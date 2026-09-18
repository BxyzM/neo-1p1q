"""
Guards that the circuits actually re-upload the inputs once per layer.

Every circuit here calls encode() inside build()'s per-layer loop, which is
what makes this data reuploading rather than a single up-front encoding. That
is easy to break by hoisting encode() out of the loop -- the circuit would
still train and still look correct at num_layers=1, which is the configured
default in base.yaml. These tests pin the behaviour by counting the encoding
gates in the tape, and by checking the tape actually changes with the input.
"""
import unittest

import pennylane as qml
import pennylane.numpy as np

from quantum.circuits.base import CircuitWeights
from quantum.circuits import registry

WIRES = 4
BATCH = 3
CIRCUIT_TYPES = ('normal', 'experimental_001', 'experimental_002', 'experimental_003')


def is_encoding(op):
    """True for a gate whose angle comes from the batched inputs.

    Gate *type* cannot be used to identify encodings: rotate() also emits RX,
    with a scalar trainable angle. What distinguishes an encoding is that its
    parameter carries the batch dimension, since it is computed from inputs.
    Restricting to single-wire gates then excludes the dR-conditioned IsingZZ
    and IsingYY entanglers, which are input-dependent but are not the encoding.
    """
    if len(op.wires) != 1:
        return False
    return np.shape(op.parameters[0]) == (BATCH,)


def build_weights(circuit, num_layers, wires):
    """Aux weights broadcast the way resolve_aux_weights() does at runtime."""
    pairs = wires * (wires - 1) // 2
    aux = {}
    for name, default in circuit.aux_defaults.items():
        if name in getattr(circuit, 'aux_per_wire_names', ()):
            aux[name] = np.array([default] * wires, requires_grad=True)
        elif name in getattr(circuit, 'aux_per_pair_names', ()):
            aux[name] = np.array([default] * pairs, requires_grad=True)
        else:
            aux[name] = np.array(default, requires_grad=True)
    shape = circuit.rotation_shape(wires, num_layers)
    rot = np.array(
        np.random.default_rng(0).uniform(0, np.pi, size=(shape.L, shape.N, shape.R)),
        requires_grad=True,
    )
    return CircuitWeights(rot=rot, aux=aux)


def encoding_gate_count(circuit_type, num_layers, inputs=None):
    """Number of encoding rotations in one built tape."""
    circuit = registry.get(circuit_type, num_layers=num_layers)
    wires = list(range(WIRES))
    weights = build_weights(circuit, num_layers, WIRES)
    if inputs is None:
        inputs = np.array(
            np.random.default_rng(1).uniform(-0.5, 0.5, size=(BATCH, WIRES, 3)),
            requires_grad=False,
        )
    with qml.tape.QuantumTape() as tape:
        circuit.build(weights, inputs, wires)
    return sum(is_encoding(op) for op in tape.operations), tape


class TestReuploadingScalesWithDepth(unittest.TestCase):
    """Encoding gates must scale with num_layers, not be emitted once."""

    def test_encoding_repeats_every_layer(self) -> None:
        for circuit_type in CIRCUIT_TYPES:
            with self.subTest(circuit_type=circuit_type):
                one, _ = encoding_gate_count(circuit_type, num_layers=1)
                three, _ = encoding_gate_count(circuit_type, num_layers=3)
                self.assertGreater(one, 0, "no encoding gates found at all")
                self.assertEqual(
                    three, 3 * one,
                    f"{circuit_type}: {three} encoding gates at depth 3 but {one} at depth 1; "
                    "inputs are not being re-uploaded once per layer",
                )

    def test_encoding_covers_every_wire_each_layer(self) -> None:
        # 'normal' encodes twice per wire (RY then RX); the experimental
        # circuits encode once (RX on pt). Either way it is a whole number of
        # encodings per wire per layer, never a single shared upload.
        for circuit_type in CIRCUIT_TYPES:
            with self.subTest(circuit_type=circuit_type):
                count, _ = encoding_gate_count(circuit_type, num_layers=2)
                self.assertEqual(count % (2 * WIRES), 0)


class TestReuploadedValuesDependOnInput(unittest.TestCase):
    """A reuploaded tape must carry the input into every layer's encoding."""

    def test_changing_inputs_changes_later_layer_parameters(self) -> None:
        wires = list(range(WIRES))
        for circuit_type in CIRCUIT_TYPES:
            with self.subTest(circuit_type=circuit_type):
                rng = np.random.default_rng(2)
                a = np.array(rng.uniform(-0.5, 0.5, size=(BATCH, WIRES, 3)), requires_grad=False)
                b = np.array(rng.uniform(-0.5, 0.5, size=(BATCH, WIRES, 3)), requires_grad=False)
                _, tape_a = encoding_gate_count(circuit_type, num_layers=3, inputs=a)
                _, tape_b = encoding_gate_count(circuit_type, num_layers=3, inputs=b)
                encodings_a = [op for op in tape_a.operations if is_encoding(op)]
                encodings_b = [op for op in tape_b.operations if is_encoding(op)]
                # Compare the final layer's encodings: if encode() were hoisted
                # out of the loop, later layers would reuse the first upload and
                # the tails would be identical across different inputs.
                tail_a = encodings_a[-WIRES:]
                tail_b = encodings_b[-WIRES:]
                differs = any(
                    not np.allclose(qml.math.toarray(x.parameters[0]),
                                    qml.math.toarray(y.parameters[0]))
                    for x, y in zip(tail_a, tail_b)
                )
                self.assertTrue(
                    differs,
                    f"{circuit_type}: last-layer encoding identical for different inputs",
                )


if __name__ == '__main__':
    unittest.main()
