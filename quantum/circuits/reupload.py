"""Same-particle reuploading and commuting Z/ZZ readouts for the depth study.

This is the polarization scan's extension of 1P1Q, not the successive-particle
blocks used by VQCCircuit. Quantum evolution and probabilities are PennyLane's.
"""

import math
from itertools import combinations

import pennylane as qml
import torch
from torch import nn

from .base import CircuitWeights
from .vqc import VQCCircuit
from helpers.depth_config import READOUTS


class ReuploadCircuit(VQCCircuit):
    """Reuse the existing ordered CNOT ring and Rot(0, ry, rz), RX gates."""

    def encode(self, weights, inputs, layer, wires):
        # Inputs are already (pt/jet_pt * delta_eta, pt/jet_pt * delta_phi).
        # Every layer addresses the SAME particles; there is no layer offset.
        angles = weights.aux["angle_scale"] * inputs
        for particle, wire in enumerate(wires):
            qml.RY(angles[:, particle, 0], wires=wire)
            qml.RX(angles[:, particle, 1], wires=wire)

    def measure(self, weights, wires):
        # Every Z_i and Z_i Z_j is diagonal in this one measurement basis.
        return qml.probs(wires=wires)


class ReuploadingClassifier(nn.Module):
    """PennyLane default.qubit/backprop, with one BCE logit per input jet.

    H = Z0, sum_i a_i Z_i, or sum_i a_i Z_i + sum_{i<j} b_ij Z_i Z_j.
    Computing <H> from PennyLane probabilities keeps all readout coefficients
    differentiable without constructing or evolving a custom state vector.
    """

    def __init__(self, wires, layers, readout, seed, input_angle_scale, precision):
        super().__init__()
        if type(wires) is not int or wires < 2:
            raise ValueError("The ordered CNOT ring requires at least two wires")
        if type(layers) is not int or not 1 <= layers <= 128:
            raise ValueError("layers must be an integer in [1, 128]")
        if readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}")
        if precision not in ("float32", "float64"):
            raise ValueError("precision must be float32 or float64")
        if not math.isfinite(input_angle_scale) or input_angle_scale <= 0:
            raise ValueError("input_angle_scale must be positive and finite")
        self.wires = wires
        self.layers = layers
        self.readout = readout
        self.input_angle_scale = input_angle_scale
        dtype = getattr(torch, precision)
        # Match polarization's legacy_zero_bias RNG, including drawing the
        # deformation AFTER rotations. The stream is float32 before casting.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.rotations = nn.Parameter(
            (torch.rand((layers, wires, 3), generator=generator) * math.pi).to(dtype)
        )
        self.deformation = nn.Parameter(
            (torch.rand((), generator=generator) * math.pi).to(dtype)
        )
        self.bias = nn.Parameter(torch.zeros((), dtype=dtype))
        self.pairs = tuple(combinations(range(wires), 2)) if readout == "all_z_zz" else ()
        terms = ((0,),) if readout == "z0" else tuple((i,) for i in range(wires)) + self.pairs
        basis = torch.arange(2 ** wires)
        signs = []
        for term in terms:
            parity = torch.zeros_like(basis)
            for wire in term:
                parity ^= (basis >> (wires - 1 - wire)) & 1
            signs.append(1 - 2 * parity)
        self.register_buffer("signs", torch.stack(signs, dim=1).to(dtype), persistent=False)
        if readout == "z0":
            self.register_parameter("readout_weights", None)
        else:
            initial = torch.zeros(len(terms), dtype=dtype)
            initial[0] = 1.0
            self.readout_weights = nn.Parameter(initial)

        implementation = ReuploadCircuit(num_layers=layers)
        device = qml.device("default.qubit", wires=wires, shots=None, seed=seed)

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(rotations, angle_scale, inputs):
            weights = CircuitWeights(rotations, {"angle_scale": angle_scale})
            return implementation.build(weights, inputs, list(range(wires)))

        self.qnode = circuit

    @property
    def deformation_factor(self):
        return 1 + 2 * math.pi * torch.sigmoid(self.deformation)

    def forward(self, inputs):
        if inputs.ndim != 3 or tuple(inputs.shape[1:]) != (self.wires, 2):
            raise ValueError(f"Expected (batch, {self.wires}, 2) cached base angles")
        probabilities = self.qnode(
            self.rotations, self.deformation_factor * self.input_angle_scale, inputs
        )
        # default.qubit can promote intermediate state precision. Record the
        # actual probability dtype in run metadata; keep the readout dtype
        # explicit instead of asserting the state was complex64.
        self.probability_dtype = str(probabilities.dtype)
        expectations = probabilities.to(self.rotations.dtype) @ self.signs
        score = expectations[:, 0] if self.readout_weights is None else expectations @ self.readout_weights
        return score + self.bias

    def optimizer_groups(self, weight_decay):
        classical = [self.bias]
        if self.readout_weights is not None:
            classical.append(self.readout_weights)
        return [
            {"params": [self.rotations, self.deformation], "weight_decay": 0.0},
            {"params": classical, "weight_decay": weight_decay},
        ]


class JaxReuploadingClassifier:
    """Functional JAX version of the same PennyLane circuit and readout.

    Torch supplies only the reference CPU initialization stream/sign table;
    no Torch simulation, differentiation or CUDA allocations occur here.
    The caller enables JAX x64 before constructing arrays, so default.qubit
    can retain its promoted state precision even with float32 parameters.
    """

    def __init__(self, wires, layers, readout, seed, input_angle_scale, precision, device):
        import jax
        import jax.numpy as jnp

        template = ReuploadingClassifier(wires, layers, readout, seed, input_angle_scale, precision)
        self.params = {name: jax.device_put(value.detach().numpy(), device)
                       for name, value in template.named_parameters()}
        signs = jax.device_put(template.signs.numpy(), device)
        implementation = ReuploadCircuit(num_layers=1)
        wire_list = list(range(wires))

        def qnode(function):
            return qml.QNode(function, qml.device("default.qubit", wires=wires, shots=None, seed=seed),
                             interface="jax", diff_method="backprop")

        @qnode
        def zero_state():
            return qml.state()

        @qnode
        def layer(state, rotations, angle_scale, inputs):
            # Analytic state hand-off only: no reset to |0>, decomposition,
            # measurement, normalization, or gradient stop between layers.
            qml.StatePrep(state, wires=wire_list, normalize=False, validate_norm=False)
            weights = CircuitWeights(rotations[None, ...], {"angle_scale": angle_scale})
            return implementation.build(weights, inputs, wire_list,
                                        measure_override=lambda weights, wires: qml.state())

        @qnode
        def measure(state):
            qml.StatePrep(state, wires=wire_list, normalize=False, validate_norm=False)
            return qml.probs(wires=wire_list)

        def forward(params, inputs):
            if inputs.ndim != 3 or tuple(inputs.shape[1:]) != (wires, 2):
                raise ValueError(f"Expected (batch, {wires}, 2) cached base angles")
            scale = (1 + 2 * math.pi * jax.nn.sigmoid(params["deformation"])) * input_angle_scale
            # A Python-unrolled depth-128 QNode produces an enormous XLA graph.
            # scan compiles this PennyLane layer once and preserves the state
            # and differentiability across all layers. PennyLane still owns
            # state initialization, every gate, and final probabilities.
            initial = jnp.broadcast_to(zero_state(), (inputs.shape[0], 2 ** wires))

            def evolve(state, rotations):
                return layer(state, rotations, scale, inputs), None

            state, _ = jax.lax.scan(evolve, initial, params["rotations"])
            probabilities = measure(state)
            # Dtype is static under tracing; record what PennyLane produced,
            # not an assumed state precision. Never store traced array values.
            self.probability_dtype = str(probabilities.dtype)
            expectations = jnp.matmul(probabilities.astype(params["rotations"].dtype), signs,
                                      precision=jax.lax.Precision.HIGHEST)
            score = expectations[:, 0] if readout == "z0" else jnp.matmul(
                expectations, params["readout_weights"], precision=jax.lax.Precision.HIGHEST)
            return score + params["bias"]

        self.qnode = layer
        self.measure_qnode = measure
        self.forward = forward
