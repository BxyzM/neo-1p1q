"""
VQC circuit: the variational classifier ansatz migrated from
QuantumClassifier._vqc_circuit into the Circuit protocol.
Author: Aritra Bal (ETP)
Date: 2026-09-09
"""
from typing import List, Union

import pennylane as qml
import pennylane.numpy as np

from helpers.utils import getIndex
from .base import CircuitBase, CircuitWeights


def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Sigmoid activation function.

    qml.math dispatches most ops correctly for both the autograd and jax
    interfaces, but its exp does not trace correctly under either one in
    this PennyLane version -- use each interface's own exp instead (see
    the identical logaddexp branch in quantum/math_functions.py).
    """
    if qml.math.get_interface(x) == 'jax':
        import jax.numpy as jnp
        exp = jnp.exp(-x)
    else:
        exp = np.exp(-x)
    return 1 / (1 + exp)


class VQCCircuit(CircuitBase):
    """
    Angle-encoded variational classifier: per layer, encode (eta, phi, pt) as
    rotations, entangle with a CNOT ring, then apply a trainable RZ/RY/RX per
    wire. Measured as the expval of a weighted Hamiltonian over all wires.
    """

    operations_per_qubit = 3  # RZ, RY, RX per wire per layer
    aux_defaults = {'scale_factor': 1.0, 'bias': 0.1, 'hamiltonian_coeffs': 0.1}
    aux_per_wire_names = ('hamiltonian_coeffs',)  # one independent trainable coefficient per wire

    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        self.index = {
            'eta': getIndex('particle', 'eta'),
            'phi': getIndex('particle', 'phi'),
            'pt':  getIndex('particle', 'pt'),
        }

    def encode(self, weights: CircuitWeights, inputs: np.ndarray, layer: int, wires: List[int]) -> None:
        n_wires = len(wires)
        sf = 2 * np.pi * sigmoid(weights.aux['scale_factor']) + 1
        for w in wires:
            particle = w
            zenith = inputs[:, particle, self.index['eta']]
            azimuth = inputs[:, particle, self.index['phi']]
            radius = inputs[:, particle, self.index['pt']]
            qml.RY(sf * radius * zenith, wires=w)
            qml.RX(sf * radius * azimuth, wires=w)

    def entangle(self, wires: List[int]) -> None:
        n_wires = len(wires)
        for w in wires:
            qml.CNOT(wires=[w, (w + 1) % n_wires])

    def rotate(self, weights: CircuitWeights, layer: int, wires: List[int]) -> None:
        rot_z, rot_y, rot_x = (weights.rot[layer, :, i] for i in range(3))
        for rz, ry, rx, w in zip(rot_z, rot_y, rot_x, wires):
            qml.Rot(0., ry, rz, wires=w)
            qml.RX(rx, wires=w)

    def measure(self, weights: CircuitWeights, wires: List[int]) -> qml.measurements.ExpectationMP:
        obs = [qml.PauliZ(i) for i in wires]
        coeffs = weights.aux['hamiltonian_coeffs']
        if len(coeffs) != len(wires):
            raise ValueError(
                f"hamiltonian_coeffs has {len(coeffs)} entries, circuit has {len(wires)} wires"
            )
        return qml.expval(qml.Hamiltonian(coeffs, obs))


def ring_pairs(n_wires: int) -> List[tuple]:
    """Nearest-neighbour ring pairs (w, (w+1) % n_wires), same order as entangle()."""
    return [(w, (w + 1) % n_wires) for w in range(n_wires)]


def pair_list(n_wires: int) -> List[tuple]:
    """All unordered wire pairs (i, j), i < j."""
    return [(i, j) for i in range(n_wires) for j in range(i + 1, n_wires)]


class VQCExperimental001(VQCCircuit):
    """
    VQCCircuit plus a dR-conditioned IsingZZ ring after the CNOT ring, and a
    readout extended from n trainable Z_i terms to n Z_i plus n*(n-1)/2
    trainable Z_i Z_j pairwise terms (pairwise_coeffs).
    """

    aux_defaults = {
        **VQCCircuit.aux_defaults,
        'dr_scale': 1.0,
        # One independent trainable coefficient per unordered wire pair, same
        # broadcast treatment as hamiltonian_coeffs/dr_scale -- see
        # aux_per_pair_names.
        'pairwise_coeffs': 0.1,
    }
    aux_per_wire_names = ('hamiltonian_coeffs', 'dr_scale')
    aux_per_pair_names = ('pairwise_coeffs',)

    def build(
        self, weights: CircuitWeights, inputs: np.ndarray, wires: List[int], measure_override=None,
    ) -> qml.measurements.ExpectationMP:
        dr_angle = weights.aux['dr_scale']
        for layer in range(self.num_layers):
            self.encode(weights, inputs, layer=layer, wires=wires)
            self.entangle(wires)
            for k, (i, j) in enumerate(ring_pairs(len(wires))):
                d_eta = inputs[:, i, self.index['eta']] - inputs[:, j, self.index['eta']]
                d_phi = inputs[:, i, self.index['phi']] - inputs[:, j, self.index['phi']]
                delta_r = qml.math.sqrt(d_eta * d_eta + d_phi * d_phi + 1e-12)
                qml.IsingZZ(dr_angle[k] * delta_r, wires=[i, j])
            self.rotate(weights, layer=layer, wires=wires)
        measure = measure_override or self.measure
        return measure(weights, wires)

    def measure(self, weights: CircuitWeights, wires: List[int]) -> qml.measurements.ExpectationMP:
        n_wires = len(wires)
        pairs = pair_list(n_wires)
        z_coeffs = weights.aux['hamiltonian_coeffs']
        if 'pairwise_coeffs' not in weights.aux:
            raise ValueError(
                "VQCExperimental001 requires 'pairwise_coeffs' in aux weights "
                f"({len(pairs)} independent trainable coefficients, one per unordered wire "
                "pair) -- missing from the provided weights. Add it via aux_weights, or check "
                "circuit_type if resuming/evaluating a checkpoint trained with a different circuit."
            )
        pairwise_coeffs = weights.aux['pairwise_coeffs']
        if len(z_coeffs) != n_wires:
            raise ValueError(f"hamiltonian_coeffs has {len(z_coeffs)} entries, circuit has {n_wires} wires")
        if len(pairwise_coeffs) != len(pairs):
            raise ValueError(f"pairwise_coeffs has {len(pairwise_coeffs)} entries, circuit has {len(pairs)} wire pairs")
        obs = [qml.PauliZ(i) for i in wires] + [qml.PauliZ(i) @ qml.PauliZ(j) for i, j in pairs]
        coeffs = qml.math.stack(list(z_coeffs) + list(pairwise_coeffs))
        return qml.expval(qml.Hamiltonian(coeffs, obs))
