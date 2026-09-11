"""
Save and verify the configuration, implementation, and final weights of a run.

Author: Aritra Bal (ETP)
Date: 2026-09-10
"""

import hashlib
import importlib.util
from pathlib import Path
import pickle
import shutil
import sys
from typing import Any
from types import ModuleType
import warnings

import numpy as np
import pennylane.numpy as qnp
from omegaconf import DictConfig, OmegaConf

from helpers.config import validate_training_config
from quantum.architectures import QuantumClassifier
from quantum.circuits.base import CircuitWeights

CIRCUIT_FILES = ('base.py', 'registry.py', 'vqc.py')


def save_circuit_snapshot(run_dir: str) -> Path:
    """Copy the circuit package used by training into the run directory.

    Args:
        run_dir: Run directory to create a ``circuits/`` subdirectory under.

    Returns:
        The created ``circuits/`` directory.
    """
    from quantum.circuits import base

    source_dir = Path(base.__file__).resolve().parent
    circuit_dir = Path(run_dir) / 'circuits'
    circuit_dir.mkdir(parents=True, exist_ok=False)
    for name in CIRCUIT_FILES:
        shutil.copy2(source_dir / name, circuit_dir / name)
    return circuit_dir


def _circuit_dir(run_dir: str | Path) -> Path:
    """Return a complete saved circuit directory or raise with missing files.

    Args:
        run_dir: Run directory expected to contain a ``circuits/`` subdirectory.

    Returns:
        The ``circuits/`` directory, verified to contain all of ``CIRCUIT_FILES``.

    Raises:
        FileNotFoundError: If any file in ``CIRCUIT_FILES`` is missing.
    """
    circuit_dir = Path(run_dir) / 'circuits'
    missing = [name for name in CIRCUIT_FILES if not (circuit_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing saved circuit files under {circuit_dir}: {', '.join(missing)}"
        )
    return circuit_dir


def load_circuit_snapshot(run_dir: str | Path) -> ModuleType:
    """Import a run's saved circuit registry independently of the global package.

    Loads ``circuits/registry.py`` (and its ``base``/``vqc`` siblings) from the
    run's own snapshot under a digest-named package, so the live repo's
    ``quantum.circuits`` implementation is never used for circuit construction.
    Reimporting the same snapshot in the same process reuses the cached module.

    Args:
        run_dir: Run directory containing a saved ``circuits/`` snapshot.

    Returns:
        The saved run's ``registry`` module.

    Raises:
        FileNotFoundError: If the saved circuit files are incomplete.
        ImportError: If the saved registry cannot be loaded.
    """
    circuit_dir = _circuit_dir(run_dir)
    digest = hashlib.sha256(
        b''.join((circuit_dir / name).read_bytes() for name in CIRCUIT_FILES)
    ).hexdigest()[:16]
    package_name = f'_saved_circuits_{digest}'
    registry_name = f'{package_name}.registry'
    if registry_name in sys.modules:
        return sys.modules[registry_name]

    package = ModuleType(package_name)
    package.__path__ = [str(circuit_dir)]
    package.__package__ = package_name
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(registry_name, circuit_dir / 'registry.py')
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load saved circuit registry from {circuit_dir}')
    saved_registry = importlib.util.module_from_spec(spec)
    sys.modules[registry_name] = saved_registry
    spec.loader.exec_module(saved_registry)
    return saved_registry


def implementation_signature(circuit_dir: str | Path) -> dict[str, str]:
    """Fingerprint the three circuit files saved with a training run.

    Args:
        circuit_dir: A run's saved ``circuits/`` directory (its parent, the
            run directory, is used to re-locate and re-verify it).

    Returns:
        Mapping from each file in ``CIRCUIT_FILES`` to its SHA-256 hex digest.

    Raises:
        FileNotFoundError: If any saved circuit file is missing.
    """
    directory = _circuit_dir(Path(circuit_dir).parent)
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in CIRCUIT_FILES
    }


def latest_checkpoint(run_dir: str | Path) -> Path:
    """Return the checkpoint with the greatest numeric epoch.

    Args:
        run_dir: Run directory containing a ``checkpoints/`` subdirectory.

    Returns:
        Path to the ``epNNNN.pickle`` file with the highest epoch number.

    Raises:
        FileNotFoundError: If no epoch checkpoint files are found.
    """
    checkpoint_dir = Path(run_dir) / 'checkpoints'
    checkpoints = []
    for path in checkpoint_dir.glob('ep*.pickle'):
        epoch = path.stem.removeprefix('ep')
        if epoch.isdigit():
            checkpoints.append((int(epoch), path))
    if not checkpoints:
        raise FileNotFoundError(f'No epoch checkpoints found under {checkpoint_dir}.')
    return max(checkpoints, key=lambda item: item[0])[1]


def _validate_adam_optimizer_state(optimizer: dict, path: Path, completed_epoch: int) -> None:
    """Validate a qml.AdamOptimizer checkpoint payload (autograd backend).

    Args:
        optimizer: The checkpoint's ``'optimizer'`` payload block.
        path: Checkpoint file path, used only for error messages.
        completed_epoch: Epoch recorded in the same checkpoint, used to decide
            whether accumulated Adam moments should be present yet.

    Raises:
        ValueError: If required fields are missing, hyperparameters are
            invalid, the accumulation structure doesn't match the epoch, or
            any moment is nonfinite.
    """
    required_optimizer_fields = {'name', 'stepsize', 'beta1', 'beta2', 'eps', 'accumulation'}
    if not required_optimizer_fields.issubset(optimizer):
        raise ValueError(f'Checkpoint {path} has incomplete optimizer state.')
    if not all(np.isfinite(optimizer[key]) for key in ('stepsize', 'beta1', 'beta2', 'eps')):
        raise ValueError(f'Checkpoint {path} contains invalid optimizer settings.')
    if (optimizer['stepsize'] <= 0 or optimizer['eps'] <= 0
            or not 0 <= optimizer['beta1'] < 1 or not 0 <= optimizer['beta2'] < 1):
        raise ValueError(f'Checkpoint {path} contains invalid Adam hyperparameters.')
    accumulation = optimizer['accumulation']
    if completed_epoch == 0:
        if accumulation is not None:
            raise ValueError(f'Checkpoint {path} has optimizer moments before training.')
    elif (not isinstance(accumulation, dict)
          or any(key not in accumulation for key in ('fm', 'sm', 't'))
          or not isinstance(accumulation.get('fm'), (list, tuple))
          or not isinstance(accumulation.get('sm'), (list, tuple))
          or type(accumulation['t']) is not int
          or accumulation['t'] < 1):
        raise ValueError(f'Checkpoint {path} has invalid Adam accumulation state.')
    if accumulation is not None:
        moments = [*accumulation['fm'], *accumulation['sm']]
        if not all(np.all(np.isfinite(moment)) for moment in moments):
            raise ValueError(f'Checkpoint {path} contains nonfinite Adam moments.')


def _validate_optax_optimizer_state(optimizer: dict, path: Path) -> None:
    """Validate an optax.inject_hyperparams(optax.adam) checkpoint payload (jax backend).

    optax's payload has a completely different field set from qml.AdamOptimizer's
    (name/opt_state, not name/stepsize/beta1/.../accumulation), so this is a
    separate validator rather than a branch inside the Adam one.

    Args:
        optimizer: The checkpoint's ``'optimizer'`` payload block.
        path: Checkpoint file path, used only for error messages.

    Raises:
        ValueError: If required fields are missing, the learning rate is
            invalid, or any optimizer-state leaf is nonfinite.
    """
    if not {'name', 'opt_state'}.issubset(optimizer):
        raise ValueError(f'Checkpoint {path} has incomplete optimizer state.')
    opt_state = optimizer['opt_state']
    hyperparams = getattr(opt_state, 'hyperparams', None)
    if not isinstance(hyperparams, dict) or 'learning_rate' not in hyperparams:
        raise ValueError(f'Checkpoint {path} has invalid optax hyperparameters.')
    learning_rate = hyperparams['learning_rate']
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError(f'Checkpoint {path} contains an invalid optax learning rate.')
    import jax
    leaves = jax.tree_util.tree_leaves(opt_state)
    if not leaves or not all(np.all(np.isfinite(leaf)) for leaf in leaves):
        raise ValueError(f'Checkpoint {path} contains nonfinite optax optimizer state.')


def load_training_checkpoint(
    run_dir: str | Path,
    cfg: DictConfig,
    signature: dict[str, str],
) -> tuple[Path, dict[str, Any]]:
    """Load and validate the latest resumable training checkpoint.

    Args:
        run_dir: Run directory containing the ``checkpoints/`` subdirectory.
        cfg: Resolved config the resumed run must exactly match.
        signature: Expected circuit-file hashes (from `implementation_signature`)
            the checkpoint must exactly match.

    Returns:
        Tuple of (checkpoint path, validated payload dict).

    Raises:
        FileNotFoundError: If no checkpoint exists.
        ValueError: If the checkpoint is unreadable, or its config, circuit
            signature, epoch metadata, history, or optimizer state don't
            match or are internally inconsistent, or training already
            reached ``cfg.epochs``.
    """
    path = latest_checkpoint(run_dir)
    try:
        with path.open('rb') as stream:
            payload = pickle.load(stream)
    except (EOFError, pickle.UnpicklingError) as error:
        raise ValueError(f'Cannot read checkpoint {path}.') from error

    if not isinstance(payload, dict):
        raise ValueError(f'Checkpoint {path} does not contain a training state.')
    training = payload.get('training')
    optimizer = payload.get('optimizer')
    if not isinstance(training, dict) or not isinstance(optimizer, dict) or 'weights' not in payload:
        raise ValueError(f'Checkpoint {path} is missing weights, optimizer state, or training state.')

    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if training.get('config') != config:
        raise ValueError(f'Checkpoint {path} does not match the saved configuration.')
    if training.get('implementation') != signature:
        raise ValueError(f'Checkpoint {path} does not match the saved circuit implementation.')

    completed_epoch = training.get('completed_epoch')
    filename_epoch = int(path.stem.removeprefix('ep'))
    if type(completed_epoch) is not int or completed_epoch != filename_epoch:
        raise ValueError(f'Checkpoint {path} has inconsistent epoch metadata.')
    history = training.get('history')
    if not isinstance(history, dict) or any(key not in history for key in ('train', 'val', 'auc')):
        raise ValueError(f'Checkpoint {path} has incomplete training history.')
    if type(training.get('n_decays')) is not int:
        raise ValueError(f'Checkpoint {path} has incomplete learning-rate state.')
    if (len(history['train']) != completed_epoch
            or len(history['val']) != completed_epoch + 1
            or len(history['auc']) != completed_epoch + 1):
        raise ValueError(f'Checkpoint {path} history does not match its completed epoch.')
    if not all(np.all(np.isfinite(history[key])) for key in ('train', 'val', 'auc')):
        raise ValueError(f'Checkpoint {path} contains nonfinite training history.')

    optimizer_name = optimizer.get('name')
    if optimizer_name == 'AdamOptimizer':
        _validate_adam_optimizer_state(optimizer, path, completed_epoch)
    elif optimizer_name == 'optax_adam':
        _validate_optax_optimizer_state(optimizer, path)
    else:
        raise ValueError(f'Unsupported checkpoint optimizer: {optimizer_name}.')

    if completed_epoch >= cfg.epochs:
        raise ValueError(
            f'Checkpoint epoch {completed_epoch} already reached the configured limit of {cfg.epochs}.'
        )
    return path, payload


def resolve_aux_weights(model: QuantumClassifier, aux_overrides: dict | None = None) -> dict:
    """Merge a circuit's aux defaults with overrides, broadcasting any
    circuit-declared per-wire names (see Circuit.aux_per_wire_names) from one
    scalar to one independent value per wire.

    Args:
        model: Classifier whose circuit implementation and wire count are used.
        aux_overrides: Config-supplied overrides merged on top of the circuit's
            ``aux_defaults`` (e.g. ``cfg.aux_weights``).

    Returns:
        Mapping of aux weight name to its resolved initial value: a scalar,
        or a list with one entry per wire for each name in
        ``aux_per_wire_names``.
    """
    merged = {**model._impl.aux_defaults, **(aux_overrides or {})}
    n_wires = len(model.auto_wires)
    for name in model._impl.aux_per_wire_names:
        merged[name] = [merged[name]] * n_wires
    return merged


def validate_weights(model: QuantumClassifier, weights: Any, cfg: DictConfig) -> None:
    """Reject missing, nonfinite, or incompatible structured circuit weights.

    Args:
        model: Classifier whose circuit implementation defines the expected
            rotation shape and aux weight names/shapes.
        weights: Candidate weights to validate.
        cfg: Config supplying any ``aux_weights`` overrides.

    Raises:
        ValueError: If ``weights`` isn't a `CircuitWeights`, its rotation
            shape or aux weight names/shapes don't match the circuit and
            config, or any value is nonfinite.
    """
    if not isinstance(weights, CircuitWeights):
        raise ValueError('The trained model must contain structured CircuitWeights.')
    shape = model._impl.rotation_shape(len(model.auto_wires), model.num_layers)
    expected_shape = (shape.L, shape.N, shape.R)
    if np.shape(weights.rot) != expected_shape:
        raise ValueError(f'Rotation weights have shape {np.shape(weights.rot)}; expected {expected_shape}.')
    auxiliary = resolve_aux_weights(model, dict(cfg.get('aux_weights', {})))
    if set(weights.aux) != set(auxiliary):
        raise ValueError('Auxiliary weight names do not match the saved configuration.')
    for name, initial in auxiliary.items():
        if np.shape(weights.aux[name]) != np.shape(initial):
            raise ValueError(f'Auxiliary weight shape does not match the configuration: {name}')
    if not all(np.all(np.isfinite(value)) for value in [weights.rot, *weights.aux.values()]):
        raise ValueError('Trained weights contain nonfinite values.')


def save_trained_run(
    run_dir: str, cfg: DictConfig, model: QuantumClassifier, weights: CircuitWeights,
    history: dict[str, list[float]], signature: dict[str, Any],
) -> None:
    """Write final weights with provenance only after successful training.

    Args:
        run_dir: Run directory containing ``config.yaml`` and ``circuits/``.
        cfg: Resolved config the run was trained with.
        model: Classifier used to validate ``weights`` against its circuit.
        weights: Final trained weights to certify and save.
        history: Per-epoch train/val loss and AUC history.
        signature: Circuit-file hashes the saved snapshot must still match.

    Raises:
        ValueError: If the training history is incomplete or nonfinite, the
            saved circuit files changed since training started, ``weights``
            fails validation, or the saved YAML no longer matches ``cfg``.
    """
    epochs = len(history['train'])
    if (epochs < 1 or len(history['val']) != epochs + 1 or len(history['auc']) != epochs + 1
            or not all(np.all(np.isfinite(history[key])) for key in ('train', 'val', 'auc'))):
        raise ValueError('Cannot certify final weights: incomplete training history or nonfinite metrics.')
    if implementation_signature(Path(run_dir) / 'circuits') != signature:
        raise ValueError('Saved circuit implementation changed during training.')
    validate_weights(model, weights, cfg)
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    saved_config = OmegaConf.to_container(OmegaConf.load(Path(run_dir) / 'config.yaml'), resolve=True)
    if config != saved_config:
        raise ValueError('Saved YAML differs from the configuration used for training.')
    payload = {
        'weights': weights,
        'training': {
            'config': config,
            'implementation': signature,
            'completed_epochs': epochs,
            'stop_reason': 'early_stopping' if epochs < cfg.epochs else 'epoch_limit',
        },
    }
    path = Path(run_dir) / 'trained_model.pickle'
    temporary = path.with_suffix('.pickle.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(payload, stream)
    temporary.replace(path)


def load_trained_run(config_path: str) -> tuple[DictConfig, QuantumClassifier]:
    """Load the final model beside its YAML, rejecting unverifiable run artifacts.

    Reimports the run's own snapshotted circuit implementation (not the live
    ``quantum.circuits`` package) and installs the certified weights as
    non-trainable. Emits a `RuntimeWarning` since completed training does not
    by itself establish convergence.

    Args:
        config_path: Path to the run's saved ``config.yaml``.

    Returns:
        Tuple of (loaded config, classifier with verified final weights
        installed).

    Raises:
        FileNotFoundError: If ``trained_model.pickle`` is missing.
        ValueError: If training didn't complete, or the saved config, circuit
            implementation, or weights don't match what was recorded at
            training time.
    """
    path = Path(config_path).expanduser().resolve()
    cfg = OmegaConf.load(path)
    validate_training_config(cfg)
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    model_path = path.parent / 'trained_model.pickle'
    if not model_path.is_file():
        raise FileNotFoundError(f'Missing final trained weights: {model_path}; intermediate checkpoints are not accepted.')
    with model_path.open('rb') as stream:
        payload = pickle.load(stream)
    training = payload.get('training') if isinstance(payload, dict) else None
    if not isinstance(training, dict) or training.get('completed_epochs', 0) < 1:
        raise ValueError('Missing completed-training provenance; cannot verify these final weights.')
    if training.get('config') != config:
        raise ValueError('Saved YAML does not match the configuration recorded with the trained weights.')
    circuit_dir = _circuit_dir(path.parent)
    if training.get('implementation') != implementation_signature(circuit_dir):
        raise ValueError('Saved circuit files differ from those used for training.')
    saved_registry = load_circuit_snapshot(path.parent)
    model = QuantumClassifier.from_config(cfg, circuit_registry=saved_registry)
    weights = payload.get('weights')
    validate_weights(model, weights, cfg)
    model.current_weights = CircuitWeights(
        rot=qnp.array(weights.rot, requires_grad=False),
        aux={name: qnp.array(value, requires_grad=False) for name, value in weights.aux.items()},
    )
    warnings.warn(
        f"Loaded final weights after {training['completed_epochs']} epochs "
        f"({training.get('stop_reason', 'unknown stop reason')}). "
        'Training completion does not establish convergence; inspect the validation history.',
        RuntimeWarning,
        stacklevel=2,
    )
    return cfg, model
