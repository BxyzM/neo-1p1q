"""
Verify saved configuration, circuit reconstruction, and final-weight provenance.

Author: Aritra Bal (ETP)
Date: 2026-09-10
"""

import contextlib
import io
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from omegaconf import OmegaConf
import pennylane.numpy as qnp

from helpers.config import DEFAULT_CONFIG, evaluation_config_path, load_config, save_config
from helpers.trained_run import (
    CIRCUIT_FILES,
    implementation_signature,
    latest_checkpoint,
    load_circuit_snapshot,
    load_training_checkpoint,
    load_trained_run,
    resolve_aux_weights,
    save_circuit_snapshot,
    save_trained_run,
)
from quantum.architectures import QuantumClassifier
from quantum.circuits.base import CircuitWeights
from quantum.losses import VQC_cost


class TestSavedRun(unittest.TestCase):
    """Use a nondefault rotation shape to catch train/evaluate configuration drift."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.cfg = OmegaConf.merge(OmegaConf.load(DEFAULT_CONFIG), {
            'seed': 'run', 'wires': 2, 'num_layers': 2, 'operations_per_qubit': 4,
            'shots': -1, 'device_name': 'default.qubit', 'epochs': 1,
            'save_dir': str(self.root), 'dump': str(self.root / 'results'),
            'aux_weights': {'bias': 0.25}, 'new_option': {'value': 7},
        })
        self.circuit_dir = save_circuit_snapshot(str(self.root))
        self.registry = load_circuit_snapshot(self.root)
        self.model = QuantumClassifier.from_config(self.cfg, circuit_registry=self.registry)
        self.weights = CircuitWeights(
            rot=qnp.array(np.arange(16).reshape(2, 2, 4) / 20),
            aux={
                name: qnp.array(value)
                for name, value in resolve_aux_weights(self.model, dict(self.cfg.aux_weights)).items()
            },
        )
        self.config_path = self.root / 'config.yaml'
        save_config(self.cfg, str(self.config_path))
        self.history = {'train': [0.5], 'val': [0.6, 0.4], 'auc': [0.5, 0.7]}
        save_trained_run(str(self.root), self.cfg, self.model, self.weights,
                         self.history, implementation_signature(self.circuit_dir))

    def _checkpoint_payload(self) -> dict:
        """Return a valid epoch-zero checkpoint for this saved run."""
        return {
            'weights': self.weights,
            'optimizer': {
                'name': 'AdamOptimizer', 'stepsize': self.cfg.lr,
                'beta1': 0.9, 'beta2': 0.99, 'eps': 1e-8, 'accumulation': None,
            },
            'training': {
                'config': OmegaConf.to_container(self.cfg, resolve=True),
                'implementation': implementation_signature(self.circuit_dir),
                'completed_epoch': 0,
                'history': {'train': [], 'val': [0.5], 'auc': [0.5]},
                'n_decays': 0,
            },
        }

    def test_round_trip_preserves_nondefault_circuit_and_scores(self) -> None:
        with self.assertWarnsRegex(RuntimeWarning, 'does not establish convergence'):
            cfg, model = load_trained_run(str(self.config_path))
        self.assertEqual(cfg, self.cfg)
        self.assertEqual(model.current_weights.rot.shape, (2, 2, 4))
        self.assertFalse(model.current_weights.rot.requires_grad)
        inputs = qnp.array(np.arange(24).reshape(2, 4, 3) / 100, requires_grad=False)
        for loss_type in ('MSE', 'BCE'):
            expected = VQC_cost(self.weights, inputs, self.model.circuit, qnp.array([0, 1]), True, loss_type)
            actual = VQC_cost(model.current_weights, inputs, model.circuit, qnp.array([0, 1]), True, loss_type)
            np.testing.assert_allclose(actual[0], expected[0])
            np.testing.assert_allclose(actual[1], expected[1])

    def test_missing_final_weights_do_not_fall_back_to_checkpoints(self) -> None:
        checkpoints = self.root / 'checkpoints'
        checkpoints.mkdir()
        (self.root / 'trained_model.pickle').rename(checkpoints / 'ep01.pickle')
        with self.assertRaisesRegex(FileNotFoundError, 'intermediate checkpoints are not accepted'):
            load_trained_run(str(self.config_path))

    def test_changed_yaml_is_rejected(self) -> None:
        self.cfg.operations_per_qubit = 3
        save_config(self.cfg, str(self.config_path))
        with self.assertRaisesRegex(ValueError, 'does not match'):
            load_trained_run(str(self.config_path))

    def test_invalid_or_unverifiable_weights_are_rejected(self) -> None:
        path = self.root / 'trained_model.pickle'
        original = path.read_bytes()
        for problem in ('legacy', 'source', 'shape', 'aux', 'nonfinite'):
            with self.subTest(problem=problem):
                payload = pickle.loads(original)
                if problem == 'legacy':
                    del payload['training']
                elif problem == 'source':
                    payload['training']['implementation'] = {}
                elif problem == 'shape':
                    payload['weights'].rot = qnp.zeros((2, 2, 3))
                elif problem == 'aux':
                    del payload['weights'].aux['bias']
                else:
                    payload['weights'].rot[0, 0, 0] = np.nan
                path.write_bytes(pickle.dumps(payload))
                with self.assertRaises(ValueError):
                    load_trained_run(str(self.config_path))

    def test_run_directory_can_move(self) -> None:
        moved = self.root / 'moved'
        moved.mkdir()
        for name in ('config.yaml', 'trained_model.pickle'):
            (self.root / name).rename(moved / name)
        self.circuit_dir.rename(moved / 'circuits')
        with self.assertWarns(RuntimeWarning):
            _, model = load_trained_run(str(moved / 'config.yaml'))
        np.testing.assert_array_equal(model.current_weights.rot, self.weights.rot)

    def test_legacy_unseeded_run_remains_loadable(self) -> None:
        del self.cfg.random_seed
        save_config(self.cfg, str(self.config_path))
        model_path = self.root / 'trained_model.pickle'
        payload = pickle.loads(model_path.read_bytes())
        del payload['training']['config']['random_seed']
        model_path.write_bytes(pickle.dumps(payload))

        with self.assertWarns(RuntimeWarning):
            cfg, model = load_trained_run(str(self.config_path))
        self.assertNotIn('random_seed', cfg)
        np.testing.assert_array_equal(model.current_weights.rot, self.weights.rot)

    def test_missing_circuit_setting_is_rejected(self) -> None:
        del self.cfg.operations_per_qubit
        with self.assertRaisesRegex(ValueError, 'Missing circuit settings: operations_per_qubit'):
            QuantumClassifier.from_config(self.cfg)

    def test_incomplete_training_cannot_certify_weights(self) -> None:
        for history in ({'train': [], 'val': [0.5], 'auc': [0.5]},
                        {'train': [np.nan], 'val': [0.5, 0.4], 'auc': [0.5, 0.6]}):
            with self.subTest(history=history), self.assertRaisesRegex(ValueError, 'incomplete training history'):
                save_trained_run(str(self.root), self.cfg, self.model, self.weights,
                                 history, implementation_signature(self.circuit_dir))

    def test_saved_registry_is_used_instead_of_global_registry(self) -> None:
        with patch('quantum.architectures.registry.get', side_effect=AssertionError('global registry used')):
            with self.assertWarns(RuntimeWarning):
                _, model = load_trained_run(str(self.config_path))
        self.assertTrue(model._impl.__class__.__module__.startswith('_saved_circuits_'))

    def test_missing_saved_circuit_file_is_rejected(self) -> None:
        for name in CIRCUIT_FILES:
            with self.subTest(name=name):
                path = self.circuit_dir / name
                contents = path.read_bytes()
                path.unlink()
                try:
                    with self.assertRaisesRegex(FileNotFoundError, name):
                        load_trained_run(str(self.config_path))
                finally:
                    path.write_bytes(contents)

    def test_resume_reads_saved_settings_before_writing_yaml(self) -> None:
        import train

        del self.cfg.random_seed
        save_config(self.cfg, str(self.config_path))
        incoming = OmegaConf.merge(self.cfg, {
            'save_dir': str(self.root.parent), 'seed': self.root.name,
            'resume': True, 'wires': 99, 'new_option': {'value': 99},
        })
        checkpoint_dir = self.root / 'checkpoints'
        checkpoint_dir.mkdir()
        (checkpoint_dir / 'ep0000.pickle').write_bytes(
            pickle.dumps(self._checkpoint_payload())
        )
        wandb = MagicMock()
        wandb.run.id = 'offline-test'
        with patch.object(train, 'wandb', wandb), patch.object(train, 'logger'), \
                patch.object(QuantumClassifier, 'from_config', side_effect=RuntimeError('stop before training')):
            with self.assertRaisesRegex(RuntimeError, 'stop before training'):
                train.main(incoming)
        saved = OmegaConf.load(self.config_path)
        self.assertEqual(saved.wires, 2)
        self.assertEqual(saved.new_option.value, 7)
        self.assertFalse(saved.resume)
        self.assertEqual(wandb.init.call_args.kwargs['project'], 'neo1P1Q')
        self.assertEqual(wandb.init.call_args.kwargs['name'], 'run')
        for path_key in ('data_dir', 'save_dir', 'dump'):
            self.assertNotIn(path_key, wandb.init.call_args.kwargs['config'])

    def test_latest_checkpoint_uses_numeric_epoch(self) -> None:
        checkpoint_dir = self.root / 'checkpoints'
        checkpoint_dir.mkdir()
        for name in ('ep9.pickle', 'ep10.pickle', 'epinvalid.pickle'):
            (checkpoint_dir / name).touch()
        self.assertEqual(latest_checkpoint(self.root).name, 'ep10.pickle')

    def test_checkpoint_provenance_is_required(self) -> None:
        checkpoint_dir = self.root / 'checkpoints'
        checkpoint_dir.mkdir()
        (checkpoint_dir / 'ep0000.pickle').write_bytes(
            pickle.dumps({'weights': self.weights})
        )
        with self.assertRaisesRegex(ValueError, 'missing weights, optimizer state, or training state'):
            load_training_checkpoint(
                self.root, self.cfg, implementation_signature(self.circuit_dir)
            )

    def test_checkpoint_circuit_mismatch_is_rejected(self) -> None:
        checkpoint_dir = self.root / 'checkpoints'
        checkpoint_dir.mkdir()
        checkpoint = self._checkpoint_payload()
        checkpoint['training']['implementation'] = {}
        (checkpoint_dir / 'ep0000.pickle').write_bytes(pickle.dumps(checkpoint))
        with self.assertRaisesRegex(ValueError, 'saved circuit implementation'):
            load_training_checkpoint(
                self.root, self.cfg, implementation_signature(self.circuit_dir)
            )

    def test_resume_across_backends_is_rejected(self) -> None:
        """A jax-backend checkpoint cannot resume under an autograd-configured run
        (or vice versa) -- caught by the existing exact-config-equality check,
        confirmed here rather than assumed."""
        checkpoint_dir = self.root / 'checkpoints'
        checkpoint_dir.mkdir()
        checkpoint = self._checkpoint_payload()
        checkpoint['training']['config']['backend'] = 'jax'
        (checkpoint_dir / 'ep0000.pickle').write_bytes(pickle.dumps(checkpoint))
        autograd_cfg = OmegaConf.merge(self.cfg, {'backend': 'autograd'})
        with self.assertRaisesRegex(ValueError, 'does not match the saved configuration'):
            load_training_checkpoint(
                self.root, autograd_cfg, implementation_signature(self.circuit_dir)
            )

    def test_training_entry_point_saves_effective_options_and_final_weights(self) -> None:
        import train
        import evaluate

        argv = ['train.py', '--config', str(self.config_path), 'seed=entrypoint',
                'batch_size=2', 'new_option.value=11', 'added_option=${wires}']
        with patch('sys.argv', argv):
            cfg = load_config()
        batches = [
            (qnp.array(np.arange(24).reshape(2, 4, 3) / 100), np.array([0, 1])),
            (qnp.array(np.arange(12).reshape(1, 4, 3) / 100), np.array([1])),
        ]
        wandb = MagicMock()
        wandb.run.id = 'offline-test'
        figure, axes = MagicMock(), MagicMock()
        with patch.object(train, 'wandb', wandb), patch.object(train, 'logger'), \
                patch.object(train.cr, 'OneP1QDataLoader', return_value=batches) as loader, \
                patch.object(train.glob, 'glob', return_value=['unused.h5']), \
                patch.object(QuantumClassifier, 'print_training_params'), \
                patch.object(train.plt, 'subplots', return_value=(figure, axes)):
            train.main(cfg)
            saved = self.root / 'entrypoint' / '42' / 'config.yaml'
            reloaded = OmegaConf.load(saved)
            self.assertEqual(reloaded.batch_size, 2)
            self.assertEqual(reloaded.new_option.value, 11)
            self.assertEqual(reloaded.added_option, 2)
            self.assertNotIn('${', saved.read_text())
            with self.assertWarns(RuntimeWarning):
                evaluate.main(str(saved))
        self.assertEqual(loader.call_count, 3)
        self.assertTrue(all(call.kwargs['seed'] == 42 for call in loader.call_args_list))
        with (self.root / 'results' / 'entrypoint' / '42' / 'test_results.pickle').open('rb') as stream:
            results = pickle.load(stream)
        self.assertEqual(len(results['scores']), 3)
        self.assertTrue(np.isfinite(results['auc']))

    def test_new_training_requires_seed_and_claims_its_directory(self) -> None:
        import train

        cfg = OmegaConf.merge(self.cfg, {
            'save_dir': str(self.root), 'seed': 'new', 'random_seed': 7,
        })
        destination = self.root / 'new' / '7'
        destination.mkdir(parents=True)
        with self.assertRaisesRegex(FileExistsError, str(destination)):
            train.main(cfg)

        del cfg.random_seed
        cfg.seed = 'unseeded'
        with self.assertRaisesRegex(ValueError, 'require an integer random_seed'):
            train.main(cfg)


class TestEvaluationCLI(unittest.TestCase):
    """Evaluation selects saved files without permitting configuration overrides."""

    def test_config_and_seed_paths(self) -> None:
        self.assertEqual(evaluation_config_path(['--config', '/tmp/run/config.yaml']), '/tmp/run/config.yaml')
        self.assertEqual(evaluation_config_path(['--seed', 'run', '--model-dir', '/tmp/models']),
                         '/tmp/models/run/config.yaml')
        self.assertEqual(
            evaluation_config_path([
                '--seed', 'run', '--random-seed', '42', '--model-dir', '/tmp/models'
            ]),
            '/tmp/models/run/42/config.yaml',
        )
        expected = Path(os.path.abspath(OmegaConf.load(DEFAULT_CONFIG).save_dir)) / 'run/config.yaml'
        self.assertEqual(evaluation_config_path(['--seed', 'run']), str(expected))

    def test_overrides_and_ambiguous_sources_are_rejected(self) -> None:
        for args in ([], ['--seed', 'run', 'wires=8'],
                     ['--seed', 'run', '--config', '/tmp/config.yaml'],
                     ['--config', '/tmp/config.yaml', '--model-dir', '/tmp'],
                     ['--config', '/tmp/config.yaml', '--random-seed', '42'],
                     ['--seed', 'run', '--random-seed', '-1'],
                     ['--seed', '../run']):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                evaluation_config_path(args)
