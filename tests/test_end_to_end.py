"""
End-to-end validation of training, checkpoint resume, and inference.

Includes regression tests for the training, validation, and inference paths.
Author: Aritra Bal (ETP)
Date: 2026-09-10
"""
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

import numpy as onp
import pennylane as qml
import pennylane.numpy as np

import glob

import case_reader as cr
import helpers.utils as ut
import quantum.architectures as arch
import quantum.losses as loss
from helpers.trained_run import latest_checkpoint, resolve_aux_weights
from quantum.circuits.base import CircuitWeights

try:
    import optax
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

_JETCLASS_DIR = '/ceph/abal/JetClass'


class _FixedBatches:
    """A minimal (data, labels) iterable standing in for a real DataLoader."""

    def __init__(self, data_batches, label_batches):
        self._data = data_batches
        self._labels = label_batches

    def __iter__(self):
        return iter(zip(self._data, self._labels))

    def __len__(self):
        return len(self._data)


def _make_batches(n_batches: int, batch_size: int, n_qubits: int, n_layers: int) -> _FixedBatches:
    """n_batches batches with alternating labels, so roc_auc_score always sees both classes."""
    data = [
        np.array(onp.random.uniform(-1, 1, size=(batch_size, n_qubits * n_layers, 3)))
        for _ in range(n_batches)
    ]
    labels = [onp.array([i % 2] * batch_size) for i in range(n_batches)]
    return _FixedBatches(data, labels)


def _build_trainer(save_dir: str, batch_size: int, epochs: int, n_qubits: int = 4, n_layers: int = 1):
    """Construct a QuantumClassifier + QuantumTrainer pair, ready to train."""
    onp.random.seed(0)
    vqc = arch.QuantumClassifier(
        wires=n_qubits, shots=None, dev_name='default.qubit',
        layers=n_layers, backend_name='autograd', test=False,
    )
    vqc.set_circuit('normal', operations_per_qubit=3)

    shape = vqc._impl.rotation_shape(n_qubits, n_layers)
    rot = np.array(onp.random.uniform(0, np.pi, size=(shape.L, shape.N, shape.R)), requires_grad=True)
    aux = {k: np.array(v, requires_grad=True) for k, v in resolve_aux_weights(vqc).items()}
    init_weights = CircuitWeights(rot=rot, aux=aux)

    optimizer = qml.AdamOptimizer(stepsize=0.05)
    trainer = arch.QuantumTrainer(
        model=vqc, lr=0.05, optimizer=optimizer, loss_fn=loss.VQC_cost,
        save=True, train_max_n=batch_size * 2, valid_max_n=batch_size * 2,
        epochs=epochs, wandb=None, loss_type='MSE',
        checkpoint_config={
            'epochs': epochs, 'wires': n_qubits, 'num_layers': n_layers,
        },
        circuit_signature={'test': 'fixed'},
        init_weights=init_weights, batch_size=batch_size, logger=__import__('loguru').logger,
    )
    trainer.set_directories(save_dir)
    return vqc, trainer


def _build_jax_trainer(save_dir: str, batch_size: int, epochs: int, n_qubits: int = 4, n_layers: int = 1):
    """Construct a jax-backend QuantumClassifier + QuantumTrainer pair, ready to train."""
    onp.random.seed(0)
    vqc = arch.QuantumClassifier(
        wires=n_qubits, shots=None, dev_name='default.qubit',
        layers=n_layers, backend_name='jax', test=False,
    )
    vqc.set_circuit('normal', operations_per_qubit=3)

    shape = vqc._impl.rotation_shape(n_qubits, n_layers)
    rot = np.array(onp.random.uniform(0, np.pi, size=(shape.L, shape.N, shape.R)), requires_grad=True)
    aux = {k: np.array(v, requires_grad=True) for k, v in resolve_aux_weights(vqc).items()}
    init_weights = CircuitWeights(rot=rot, aux=aux)

    optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=0.05)
    trainer = arch.QuantumTrainer(
        model=vqc, lr=0.05, optimizer=optimizer, loss_fn=loss.VQC_cost,
        save=True, train_max_n=batch_size * 2, valid_max_n=batch_size * 2,
        epochs=epochs, wandb=None, loss_type='MSE',
        checkpoint_config={
            'epochs': epochs, 'wires': n_qubits, 'num_layers': n_layers,
        },
        circuit_signature={'test': 'fixed'},
        init_weights=init_weights, batch_size=batch_size, logger=__import__('loguru').logger,
    )
    trainer.set_directories(save_dir)
    return vqc, trainer


def _run_seeded_training(random_seed: int):
    """Run one finite-shot epoch and inference from an isolated random stream."""
    vqc = arch.QuantumClassifier(
        wires=2,
        shots=50,
        dev_name='lightning.qubit',
        layers=1,
        backend_name='autograd',
        random_seed=random_seed,
    )
    vqc.set_circuit('normal', operations_per_qubit=3, diff_method='parameter-shift')
    shape = vqc._impl.rotation_shape(2, 1)
    initial = onp.random.default_rng(random_seed).uniform(
        0, onp.pi, size=(shape.L, shape.N, shape.R)
    )
    weights = CircuitWeights(
        rot=np.array(initial, requires_grad=True),
        aux={
            name: np.array(value, requires_grad=True)
            for name, value in resolve_aux_weights(vqc).items()
        },
    )
    trainer = arch.QuantumTrainer(
        model=vqc,
        lr=0.05,
        optimizer=qml.AdamOptimizer(stepsize=0.05),
        loss_fn=loss.VQC_cost,
        save=False,
        epochs=1,
        wandb=None,
        loss_type='MSE',
        init_weights=weights,
        batch_size=2,
        logger=__import__('loguru').logger,
    )
    data = np.array([
        [[0.1, 0.2, 0.3], [0.2, -0.1, 0.4]],
        [[-0.2, 0.3, 0.5], [0.4, 0.1, 0.2]],
    ])
    labels = np.array([0, 1])
    batches = _FixedBatches([data], [labels])
    with tempfile.TemporaryDirectory() as directory:
        trainer.set_directories(directory)
        history = trainer.run_training_loop(batches, batches)
        vqc.current_weights = trainer.current_weights
        _, scores, _ = vqc.run_inference(
            batches, loss_fn=loss.VQC_cost, loss_type='MSE'
        )
        return initial, trainer.current_weights, history, scores


class TestRandomSeedReproducibility(unittest.TestCase):
    """One seed must reproduce the complete finite-shot numerical path."""

    def test_same_seed_reproduces_training_and_inference(self) -> None:
        first = _run_seeded_training(42)
        second = _run_seeded_training(42)

        onp.testing.assert_array_equal(first[0], second[0])
        onp.testing.assert_array_equal(first[1].rot, second[1].rot)
        for name in first[1].aux:
            onp.testing.assert_array_equal(first[1].aux[name], second[1].aux[name])
        self.assertEqual(first[2], second[2])
        onp.testing.assert_array_equal(first[3], second[3])

    def test_different_seed_changes_initialization(self) -> None:
        first = _run_seeded_training(42)
        second = _run_seeded_training(43)
        self.assertFalse(onp.array_equal(first[0], second[0]))

    def test_parallel_fresh_processes_reproduce_numerical_results(self) -> None:
        code = (
            "import pickle, sys; from pathlib import Path; import numpy as np; "
            "from tests.test_end_to_end import _run_seeded_training; "
            "initial, weights, history, scores = _run_seeded_training(42); "
            "payload = (np.asarray(initial), np.asarray(weights.rot), "
            "{k: np.asarray(v) for k, v in weights.aux.items()}, history, "
            "np.asarray(scores)); Path(sys.argv[1]).write_bytes(pickle.dumps(payload))"
        )
        environment = os.environ.copy()
        environment.update({
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1',
            'MPLCONFIGDIR': '/tmp/matplotlib',
        })
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f'result_{index}.pickle' for index in range(2)]
            processes = [
                subprocess.Popen(
                    [sys.executable, '-c', code, str(path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    env=environment,
                )
                for path in paths
            ]
            for process in processes:
                _, stderr = process.communicate(timeout=60)
                self.assertEqual(process.returncode, 0, stderr.decode())
            first, second = [pickle.loads(path.read_bytes()) for path in paths]

        onp.testing.assert_array_equal(first[0], second[0])
        onp.testing.assert_array_equal(first[1], second[1])
        for name in first[2]:
            onp.testing.assert_array_equal(first[2][name], second[2][name])
        self.assertEqual(first[3], second[3])
        onp.testing.assert_array_equal(first[4], second[4])


class TestFullTrainingLoop(unittest.TestCase):
    """D10: run_training_loop must run end to end and produce a usable checkpoint."""

    def test_runs_and_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as save_dir:
            vqc, trainer = _build_trainer(save_dir, batch_size=1, epochs=2)
            train_loader = _make_batches(2, batch_size=1, n_qubits=4, n_layers=1)
            val_loader = _make_batches(2, batch_size=1, n_qubits=4, n_layers=1)

            history = trainer.run_training_loop(train_loader, val_loader)

            self.assertEqual(len(history['train']), 2)  # one per epoch, epoch 0 is val-only
            self.assertEqual(len(history['val']), 3)     # epoch 0 (initial) + 2 training epochs
            self.assertEqual(len(history['auc']), 3)
            for v in history['train'] + history['val']:
                self.assertTrue(onp.isfinite(v))

            checkpoint_dir = os.path.join(save_dir, 'checkpoints')
            self.assertEqual(
                sorted(os.listdir(checkpoint_dir)),
                ['ep0000.pickle', 'ep0001.pickle', 'ep0002.pickle'],
            )

    def test_decay_after_minimum_epochs_then_early_stop(self) -> None:
        """Three failed post-warmup checks decay Adam before stopping training."""
        with tempfile.TemporaryDirectory() as save_dir:
            _, trainer = _build_trainer(save_dir, batch_size=1, epochs=30)
            trainer.saving = False
            trainer.min_epochs = 10
            trainer.decay_rate = 0.5
            trainer.decay_patience = 3

            def constant_iteration(
                data: np.ndarray, labels: np.ndarray, train: bool = False,
            ) -> float | tuple[float, np.ndarray]:
                if train:
                    return 0.25
                return 0.25, np.full(len(labels), 0.5)

            trainer.iteration = constant_iteration
            batches = _make_batches(2, batch_size=1, n_qubits=4, n_layers=1)
            history = trainer.run_training_loop(batches, batches)

            self.assertEqual(len(history['train']), 13)
            self.assertEqual(trainer.n_decays, 3)
            self.assertAlmostEqual(trainer.optim.stepsize, 0.05 * 0.5**3)

    def test_checkpoint_round_trip_and_resume_training(self) -> None:
        """A resumed Adam step must match an uninterrupted step exactly."""
        with tempfile.TemporaryDirectory() as save_dir, tempfile.TemporaryDirectory() as resumed_dir:
            _, trainer = _build_trainer(save_dir, batch_size=1, epochs=2)
            first = np.array(onp.random.uniform(-1, 1, size=(1, 4, 3)))
            second = np.array(onp.random.uniform(-1, 1, size=(1, 4, 3)))
            labels = onp.array([1])
            first_cost = trainer.iteration(first, labels=labels, train=True)
            trainer.current_epoch = 1
            trainer.history = {
                'train': [first_cost], 'val': [0.6, 0.5], 'auc': [0.5, 0.6],
            }
            checkpoint = trainer.save_checkpoint()
            payload = ut.Unpickle(checkpoint)

            expected_cost = trainer.iteration(second, labels=labels, train=True)
            expected_weights = trainer.current_weights

            _, resumed = _build_trainer(resumed_dir, batch_size=1, epochs=2)
            resumed.restore_checkpoint(payload)
            actual_cost = resumed.iteration(second, labels=labels, train=True)

            self.assertEqual(resumed.current_epoch, 2)
            self.assertEqual(resumed.history, payload['training']['history'])
            self.assertAlmostEqual(actual_cost, expected_cost)
            onp.testing.assert_allclose(resumed.current_weights.rot, expected_weights.rot)
            for name in expected_weights.aux:
                onp.testing.assert_allclose(
                    resumed.current_weights.aux[name], expected_weights.aux[name]
                )
            self.assertEqual(resumed.optim.t, trainer.optim.t)
            for actual, expected in zip(resumed.optim.fm, trainer.optim.fm):
                onp.testing.assert_allclose(actual, expected)
            for actual, expected in zip(resumed.optim.sm, trainer.optim.sm):
                onp.testing.assert_allclose(actual, expected)

    def test_resumed_loop_starts_at_next_epoch(self) -> None:
        """Resume must preserve history and skip completed epochs."""
        with tempfile.TemporaryDirectory() as save_dir, tempfile.TemporaryDirectory() as resumed_dir:
            _, trainer = _build_trainer(save_dir, batch_size=1, epochs=2)
            data = np.array(onp.random.uniform(-1, 1, size=(1, 4, 3)))
            first_cost = trainer.iteration(data, labels=onp.array([1]), train=True)
            trainer.current_epoch = 1
            trainer.history = {
                'train': [first_cost], 'val': [0.6, 0.5], 'auc': [0.5, 0.6],
            }
            payload = ut.Unpickle(trainer.save_checkpoint())

            _, resumed = _build_trainer(resumed_dir, batch_size=1, epochs=2)
            resumed.restore_checkpoint(payload)
            history = resumed.run_training_loop(
                _make_batches(2, 1, 4, 1), _make_batches(2, 1, 4, 1)
            )

            self.assertEqual(len(history['train']), 2)
            self.assertEqual(len(history['val']), 3)
            self.assertEqual(resumed.current_epoch, 2)
            self.assertEqual(os.listdir(os.path.join(resumed_dir, 'checkpoints')), ['ep0002.pickle'])

    def test_batched_training_and_validation_include_partial_batch(self) -> None:
        """Batch scores retain their dimension and epoch losses weight every sample equally."""
        with tempfile.TemporaryDirectory() as save_dir:
            vqc, trainer = _build_trainer(save_dir, batch_size=2, epochs=1)
            data_batches = [
                np.array(onp.random.uniform(-1, 1, size=(2, 4, 3))),
                np.array(onp.random.uniform(-1, 1, size=(1, 4, 3))),
            ]
            label_batches = [np.array([0, 1]), np.array([1])]
            train_loader = _FixedBatches(data_batches, label_batches)
            val_loader = _FixedBatches(data_batches, label_batches)

            batch_losses = []
            for data, labels in val_loader:
                batch_loss, scores = trainer.iteration(data, labels=labels, train=False)
                batch_losses.append(batch_loss)
                self.assertEqual(scores.shape, (len(labels),))
                self.assertAlmostEqual(
                    batch_loss,
                    float(np.mean((labels - scores) ** 2)),
                )
                serial_scores = np.concatenate([
                    trainer.iteration(
                        data[i:i + 1], labels=labels[i:i + 1], train=False
                    )[1]
                    for i in range(len(labels))
                ])
                onp.testing.assert_allclose(scores, serial_scores)
            expected_initial_val_loss = (
                batch_losses[0] * 2 + batch_losses[1]
            ) / 3

            history = trainer.run_training_loop(train_loader, val_loader)

            self.assertAlmostEqual(history['val'][0], expected_initial_val_loss)
            self.assertEqual(len(history['train']), 1)
            self.assertEqual(len(history['val']), 2)
            self.assertTrue(onp.isfinite(history['train'][0]))


@unittest.skipUnless(_HAS_JAX, 'jax is not installed in this environment')
class TestJaxFullTrainingLoop(unittest.TestCase):
    """backend='jax' must run end to end: train, checkpoint, resume, decay -- mirrors
    TestFullTrainingLoop's autograd coverage."""

    def test_runs_and_checkpoints_with_trained_weights(self) -> None:
        with tempfile.TemporaryDirectory() as save_dir:
            vqc, trainer = _build_jax_trainer(save_dir, batch_size=1, epochs=2)
            train_loader = _make_batches(2, batch_size=1, n_qubits=4, n_layers=1)
            val_loader = _make_batches(2, batch_size=1, n_qubits=4, n_layers=1)

            initial_rot = onp.array(trainer.current_weights.rot)
            history = trainer.run_training_loop(train_loader, val_loader)

            self.assertEqual(len(history['train']), 2)
            self.assertEqual(len(history['val']), 3)
            self.assertEqual(len(history['auc']), 3)
            for v in history['train'] + history['val']:
                self.assertTrue(onp.isfinite(v))

            checkpoint_dir = os.path.join(save_dir, 'checkpoints')
            self.assertEqual(
                sorted(os.listdir(checkpoint_dir)),
                ['ep0000.pickle', 'ep0001.pickle', 'ep0002.pickle'],
            )
            # The critical fix (P11): trained weights must actually differ from the
            # initial ones -- catches a silent current_weights/jax_params desync.
            self.assertFalse(onp.allclose(initial_rot, onp.array(trainer.current_weights.rot)))
            final_checkpoint = ut.Unpickle(latest_checkpoint(save_dir))
            onp.testing.assert_allclose(
                onp.array(final_checkpoint['weights'].rot), onp.array(trainer.current_weights.rot)
            )

    def test_decay_after_minimum_epochs_then_early_stop(self) -> None:
        """Three failed post-warmup checks decay the injected optax learning rate
        before stopping training."""
        with tempfile.TemporaryDirectory() as save_dir:
            _, trainer = _build_jax_trainer(save_dir, batch_size=1, epochs=30)
            trainer.saving = False
            trainer.min_epochs = 10
            trainer.decay_rate = 0.5
            trainer.decay_patience = 3

            def constant_iteration(
                data: np.ndarray, labels: np.ndarray, train: bool = False,
            ) -> float | tuple[float, np.ndarray]:
                if train:
                    return 0.25
                return 0.25, np.full(len(labels), 0.5)

            trainer.iteration = constant_iteration
            batches = _make_batches(2, batch_size=1, n_qubits=4, n_layers=1)
            history = trainer.run_training_loop(batches, batches)

            self.assertEqual(len(history['train']), 13)
            self.assertEqual(trainer.n_decays, 3)
            self.assertAlmostEqual(
                float(trainer.jax_opt_state.hyperparams['learning_rate']), 0.05 * 0.5**3
            )

    def test_checkpoint_round_trip_and_resume_training(self) -> None:
        """A resumed jax/optax step must match an uninterrupted step's trajectory
        exactly -- not just approximately (Adam moments/count carry over)."""
        with tempfile.TemporaryDirectory() as save_dir, tempfile.TemporaryDirectory() as resumed_dir:
            _, trainer = _build_jax_trainer(save_dir, batch_size=1, epochs=2)
            first = np.array(onp.random.uniform(-1, 1, size=(1, 4, 3)))
            second = np.array(onp.random.uniform(-1, 1, size=(1, 4, 3)))
            labels = onp.array([1])
            first_cost = trainer.iteration(first, labels=labels, train=True)
            trainer.current_epoch = 1
            trainer.history = {
                'train': [first_cost], 'val': [0.6, 0.5], 'auc': [0.5, 0.6],
            }
            checkpoint = trainer.save_checkpoint()
            payload = ut.Unpickle(checkpoint)

            expected_cost = trainer.iteration(second, labels=labels, train=True)
            expected_weights = trainer.current_weights

            _, resumed = _build_jax_trainer(resumed_dir, batch_size=1, epochs=2)
            resumed.restore_checkpoint(payload)
            actual_cost = resumed.iteration(second, labels=labels, train=True)

            self.assertEqual(resumed.current_epoch, 2)
            self.assertEqual(resumed.history, payload['training']['history'])
            self.assertAlmostEqual(actual_cost, expected_cost, places=12)
            onp.testing.assert_allclose(
                onp.array(resumed.current_weights.rot), onp.array(expected_weights.rot)
            )
            for name in expected_weights.aux:
                onp.testing.assert_allclose(
                    onp.array(resumed.current_weights.aux[name]),
                    onp.array(expected_weights.aux[name]),
                )


class TestEvaluateEquivalentInference(unittest.TestCase):
    """D10: reproduce evaluate.py's own pipeline against a real saved checkpoint."""

    def _trained_classifier(self, save_dir: str):
        """Train a tiny model and reload it the way evaluate.py does (test=False)."""
        vqc, trainer = _build_trainer(save_dir, batch_size=1, epochs=1)
        trainer.run_training_loop(
            _make_batches(2, 1, 4, 1), _make_batches(2, 1, 4, 1)
        )
        checkpoint = ut.Unpickle(latest_checkpoint(save_dir))

        vqc2 = arch.QuantumClassifier(
            wires=4, shots=None, dev_name='default.qubit',
            layers=1, backend_name='autograd', test=False,
        )
        vqc2.set_circuit('normal', operations_per_qubit=3)
        vqc2.current_weights = checkpoint['weights']
        return vqc2

    def test_run_inference_runs_without_total_batches(self) -> None:
        """
        P2 fix: QuantumClassifier(test=False) -- what evaluate.py/train.py build --
        no longer needs self.total_batches. run_inference() just walks the loader,
        so evaluate.py's real pipeline runs end to end.
        """
        with tempfile.TemporaryDirectory() as save_dir:
            vqc2 = self._trained_classifier(save_dir)
            self.assertFalse(hasattr(vqc2, 'total_batches'))

            test_loader = _make_batches(4, 1, 4, 1)
            costs, scores, labels = vqc2.run_inference(test_loader, loss_fn=loss.VQC_cost, loss_type='MSE')
            self.assertEqual(len(costs), 4)
            self.assertEqual(len(scores), 4)
            self.assertEqual(len(labels), 4)

    def test_run_inference_processes_multi_jet_chunks_one_at_a_time(self) -> None:
        """A loader that groups jets is still scored jet by jet: N jets in -> N results out."""
        with tempfile.TemporaryDirectory() as save_dir:
            vqc2 = self._trained_classifier(save_dir)

            test_loader = _make_batches(3, 4, 4, 1)  # 3 chunks x 4 jets = 12 jets
            costs, scores, labels = vqc2.run_inference(test_loader, loss_fn=loss.VQC_cost, loss_type='MSE')
            self.assertEqual(len(costs), 12)
            self.assertEqual(len(scores), 12)
            self.assertEqual(len(labels), 12)


class TestKnownPreExistingBugs(unittest.TestCase):
    """
    D10: pin down, with concrete reproductions, bugs already flagged (not fixed) in
    REFACTOR.md/AUDIT.md as pre-existing and out of this refactor's scope.
    """

    def test_trainer_rejects_raw_checkpoint_dictionary(self) -> None:
        """Trainer callers must extract the weights from a checkpoint dictionary."""
        with tempfile.TemporaryDirectory() as save_dir:
            vqc, trainer = _build_trainer(save_dir, batch_size=1, epochs=1)
            trainer.run_training_loop(_make_batches(2, 1, 4, 1), _make_batches(2, 1, 4, 1))
            resumed_init_weights = ut.Unpickle(latest_checkpoint(save_dir))
            self.assertIsInstance(resumed_init_weights, dict)  # not a CircuitWeights

            optimizer = qml.AdamOptimizer(stepsize=0.05)
            trainer2 = arch.QuantumTrainer(
                model=vqc, lr=0.05, optimizer=optimizer, loss_fn=loss.VQC_cost,
                save=False, epochs=1, init_weights=resumed_init_weights,
                batch_size=1, logger=None,
            )
            data = np.array(onp.random.uniform(-1, 1, size=(1, 4, 3)))
            with self.assertRaises(AttributeError):
                trainer2.iteration(data, labels=onp.array([1]), train=True)

@unittest.skipUnless(
    os.path.isdir(os.path.join(_JETCLASS_DIR, 'test', 'TTBar_'))
    and os.path.isdir(os.path.join(_JETCLASS_DIR, 'test', 'ZJetsToNuNu')),
    "JetClass data not available",
)
class TestBalancedJetClassLoader(unittest.TestCase):
    """CASEJetClassDataset must read equal signal/background and label them correctly."""

    def _files(self, sample):
        return sorted(glob.glob(os.path.join(_JETCLASS_DIR, 'test', sample, '*.h5')))

    def test_equal_split_and_both_labels(self) -> None:
        loader = cr.OneP1QDataLoader(
            signal_filelist=self._files('TTBar_'),
            background_filelist=self._files('ZJetsToNuNu'),
            n_signal=40, n_background=40,
            batch_size=18, input_shape=(4, 3), train=True, normalize_pt=False, seed=0,
        )
        total, counts = 0, {0: 0, 1: 0}
        first_shape = None
        batch_sizes = []
        for data, labels in loader:
            if first_shape is None:
                first_shape = tuple(data.shape[1:])
            batch_sizes.append(data.shape[0])
            for lbl in labels.tolist():
                counts[int(lbl)] += 1
            total += data.shape[0]
        self.assertEqual(total, 80)
        self.assertEqual(counts, {0: 40, 1: 40})
        self.assertEqual(first_shape, (4, 3))
        self.assertEqual(len(loader), 5)
        self.assertEqual(batch_sizes, [18, 18, 18, 18, 8])

    def test_signal_label_is_one(self) -> None:
        """A signal-only loader yields only label 1; background-only yields only label 0."""
        sig = cr.OneP1QDataLoader(
            signal_filelist=self._files('TTBar_'), background_filelist=self._files('TTBar_'),
            n_signal=20, n_background=0, batch_size=20, input_shape=(4, 3), seed=0,
        )
        labels = [int(x) for _, lbls in sig for x in lbls.tolist()]
        self.assertEqual(set(labels), {1})
        self.assertEqual(len(labels), 20)


if __name__ == '__main__':
    unittest.main()
