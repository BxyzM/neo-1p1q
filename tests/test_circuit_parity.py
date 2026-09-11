"""
D9 wiring test: QuantumClassifier.set_circuit()'s registry-based circuit must
match VQCCircuit called directly. This test used to compare the new
implementation against the pre-switchover _vqc_circuit method (D8); that
method is deleted as of D9, so it now guards the wiring in set_circuit()
instead -- the circuit body's correctness against the old implementation was
already locked down by D8's original comparison, recorded in refactored.MD.
Author: Aritra Bal (ETP)
2026-09-05
"""
import unittest

import numpy as onp
import pennylane as qml
import pennylane.numpy as np

import quantum.architectures as arch
from quantum.circuits.base import CircuitWeights
from quantum.circuits.vqc import VQCCircuit

try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


class TestVQCCircuitWiring(unittest.TestCase):
    """set_circuit()'s wired-up circuit must match a standalone VQCCircuit."""

    def setUp(self) -> None:
        onp.random.seed(0)
        self.num_layers = 2
        self.n_qubits = 4
        self.wires = list(range(self.n_qubits))
        self.weights = CircuitWeights(
            rot=np.array(
                onp.random.uniform(0, np.pi, size=(self.num_layers, self.n_qubits, 3))
            ),
            aux={
                'scale_factor': np.array(1.0),
                'bias': np.array(0.1),
                'hamiltonian_coeffs': np.array([0.1] * self.n_qubits),
            },
        )
        self.inputs = np.array(
            onp.random.uniform(-1, 1, size=(1, self.n_qubits * self.num_layers, 3))
        )
        self.qc = arch.QuantumClassifier(
            wires=self.n_qubits,
            layers=self.num_layers,
            shots=None,
            dev_name='default.qubit',
            backend_name='autograd',
        )
        self.qc.set_circuit('normal')

    def test_expval_matches(self) -> None:
        """qc.circuit (built by set_circuit) must match VQCCircuit.build() directly."""
        standalone = qml.QNode(
            VQCCircuit(num_layers=self.num_layers).build, self.qc.device, interface=self.qc.backend
        )
        wired_out = self.qc.circuit(self.weights, self.inputs)
        standalone_out = standalone(self.weights, self.inputs, self.wires)
        self.assertEqual(wired_out.shape, (1,))
        self.assertEqual(standalone_out.shape, (1,))
        self.assertAlmostEqual(float(wired_out[0]), float(standalone_out[0]), places=10)

    def test_expval_golden_value(self) -> None:
        """
        Pin the exact analytic expval for this fixed seed/weights/input. Guards
        against cross-version drift in PennyLane's gate math -- default.qubit
        with shots=None is exact linear algebra, so this constant must hold on
        any PennyLane version. Reference computed on PennyLane 0.37.0.
        """
        out = float(self.qc.circuit(self.weights, self.inputs)[0])
        self.assertAlmostEqual(out, 0.034755910654483246, places=12)


class TestSetCircuitDiffMethodValidation(unittest.TestCase):
    """set_circuit() must reject an incompatible diff_method with a clear, actionable error."""

    def test_backprop_on_finite_shot_lightning_qubit_raises_with_suggestion(self) -> None:
        qc = arch.QuantumClassifier(
            wires=2, layers=1, shots=50, dev_name='lightning.qubit', backend_name='autograd',
        )
        with self.assertRaises(ValueError) as context:
            qc.set_circuit('normal', diff_method='backprop')
        message = str(context.exception)
        self.assertIn('backprop', message)
        self.assertIn('parameter-shift', message)

    def test_parameter_shift_on_finite_shot_lightning_qubit_succeeds(self) -> None:
        qc = arch.QuantumClassifier(
            wires=2, layers=1, shots=50, dev_name='lightning.qubit', backend_name='autograd',
        )
        qc.set_circuit('normal', diff_method='parameter-shift')
        self.assertIsNotNone(qc.circuit)

@unittest.skipUnless(_HAS_JAX, 'jax is not installed in this environment')
class TestJaxBackendParity(unittest.TestCase):
    """backend='jax' must reproduce the same analytic physics as backend='autograd'."""

    def test_jax_backend_with_finite_shots_is_rejected(self) -> None:
        qc = arch.QuantumClassifier(
            wires=2, layers=1, shots=50, dev_name='default.qubit', backend_name='jax',
        )
        with self.assertRaises(ValueError) as context:
            qc.set_circuit('normal')
        message = str(context.exception)
        self.assertIn('jax', message)
        self.assertIn('analytic', message)

    def setUp(self) -> None:
        onp.random.seed(0)
        self.num_layers = 2
        self.n_qubits = 4
        self.golden_weights = CircuitWeights(
            rot=np.array(
                onp.random.uniform(0, np.pi, size=(self.num_layers, self.n_qubits, 3))
            ),
            aux={
                'scale_factor': np.array(1.0),
                'bias': np.array(0.1),
                'hamiltonian_coeffs': np.array([0.1] * self.n_qubits),
            },
        )
        self.golden_inputs = np.array(
            onp.random.uniform(-1, 1, size=(1, self.n_qubits * self.num_layers, 3))
        )
        self.golden_value = 0.034755910654483246  # same constant TestVQCCircuitWiring pins

    def _jax_classifier(self) -> arch.QuantumClassifier:
        qc = arch.QuantumClassifier(
            wires=self.n_qubits, layers=self.num_layers, shots=None,
            dev_name='default.qubit', backend_name='jax',
        )
        qc.set_circuit('normal', diff_method='backprop')
        return qc

    def test_golden_value_matches_with_batch_size_greater_than_one(self) -> None:
        """Exercise broadcast_expand with a real batch (>1) under interface='jax'."""
        qc = self._jax_classifier()
        weights = CircuitWeights(
            rot=jnp.array(onp.asarray(self.golden_weights.rot)),
            aux={k: jnp.array(onp.asarray(v)) for k, v in self.golden_weights.aux.items()},
        )
        batch_inputs = jnp.array(onp.tile(onp.asarray(self.golden_inputs), (3, 1, 1)))
        out = qc.circuit(weights, batch_inputs)
        self.assertEqual(out.shape, (3,))
        for value in out:
            self.assertAlmostEqual(float(value), self.golden_value, places=10)

    def test_hamiltonian_coeffs_gradient_matches_diff_method_pattern(self) -> None:
        """backprop/parameter-shift differentiate hamiltonian_coeffs; adjoint silently zeros it --
        same pattern already established for the autograd interface (problems.MD P9), now
        confirmed to extend to interface='jax'.
        """
        rot = jnp.array(onp.asarray(self.golden_weights.rot))
        bias = jnp.array(onp.asarray(self.golden_weights.aux['bias']))
        hc = jnp.array(onp.asarray(self.golden_weights.aux['hamiltonian_coeffs']))
        inputs = jnp.array(onp.asarray(self.golden_inputs))

        def make_cost(diff_method):
            qc = arch.QuantumClassifier(
                wires=self.n_qubits, layers=self.num_layers, shots=None,
                dev_name='default.qubit', backend_name='jax',
            )
            qc.set_circuit('normal', diff_method=diff_method)

            def cost(hamiltonian_coeffs):
                weights = CircuitWeights(
                    rot=rot, aux={'scale_factor': jnp.array(1.0), 'bias': bias,
                                  'hamiltonian_coeffs': hamiltonian_coeffs},
                )
                return jnp.sum(qc.circuit(weights, inputs))
            return cost

        grad_backprop = jax.grad(make_cost('backprop'))(hc)
        grad_param_shift = jax.grad(make_cost('parameter-shift'))(hc)
        grad_adjoint = jax.grad(make_cost('adjoint'))(hc)

        self.assertTrue(bool(jnp.all(grad_backprop != 0)))
        onp.testing.assert_allclose(
            onp.asarray(grad_backprop), onp.asarray(grad_param_shift), rtol=1e-6, atol=1e-8
        )
        onp.testing.assert_array_equal(onp.asarray(grad_adjoint), onp.zeros(self.n_qubits))


if __name__ == '__main__':
    unittest.main()
