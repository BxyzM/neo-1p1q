"""
Minimal config handling for train.py / evaluate.py.

The repo only ever needed "load one YAML, allow key=value overrides on the CLI,
and save the exact config a run used to a single reusable file". That is done
here with OmegaConf directly -- no Hydra. `cfg` stays a DictConfig, so the dot
and bracket access the rest of the code relies on is unchanged.

Author: Aritra Bal (ETP)
Date: 2026-09-10
"""
import argparse
import os
from pathlib import Path
from typing import Iterable

from omegaconf import DictConfig, OmegaConf

# Anchored to the repo root (parent of helpers/) so it resolves regardless of cwd.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(_REPO_ROOT, "configs", "base.yaml")
ORCHESTRATION_KEYS = ('number_of_runs', 'num_processes')
SUPPORTED_BACKENDS = ('autograd', 'jax')
SINGLE_THREAD_ENV = {
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
}


def positive_integer(value: str) -> int:
    """Parse a strictly positive command-line integer."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError('must be at least 1')
    return parsed


def gpu_id_list(value: str) -> list[int]:
    """Parse a comma-separated list of non-negative GPU indices, e.g. '0' or '0,1'."""
    try:
        parsed = [int(item) for item in value.split(',')]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"'{value}' is not a comma-separated list of integers"
        ) from error
    if any(gpu_id < 0 for gpu_id in parsed):
        raise argparse.ArgumentTypeError('GPU indices must be non-negative')
    return parsed


def load_config_values(config_path: str, overrides: Iterable[str] = ()) -> DictConfig:
    """Load one YAML file and merge OmegaConf dotlist overrides into it."""
    cfg = OmegaConf.load(os.path.abspath(os.path.expanduser(config_path)))
    overrides = list(overrides)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def validate_training_config(cfg: DictConfig, require_random_seed: bool = False) -> int | None:
    """Validate settings shared by direct and orchestrated training."""
    misplaced = [key for key in ORCHESTRATION_KEYS if key in cfg]
    if misplaced:
        raise ValueError(
            f"{', '.join(misplaced)} are launcher options and must not be in a training config."
        )
    min_epochs = cfg.get('min_epochs')
    decay_patience = cfg.get('decay_patience')
    decay_rate = cfg.get('decay_rate')
    if type(min_epochs) is not int or min_epochs < 1:
        raise ValueError('min_epochs must be an integer greater than or equal to 1.')
    if type(decay_patience) is not int or decay_patience < 1:
        raise ValueError('decay_patience must be an integer greater than or equal to 1.')
    if type(decay_rate) not in (int, float) or not 0 < decay_rate < 1:
        raise ValueError('decay_rate must be greater than 0 and less than 1.')
    backend = cfg.get('backend')
    if backend is not None and backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"backend='{backend}' is not supported. Use one of {SUPPORTED_BACKENDS}."
        )
    if 'random_seed' not in cfg or cfg.random_seed is None:
        if require_random_seed:
            raise ValueError('New training runs require an integer random_seed.')
        return None
    if type(cfg.random_seed) is not int or cfg.random_seed < 0:
        raise ValueError('random_seed must be an integer greater than or equal to 0.')
    return cfg.random_seed


def run_directory(root: str, seed: object, random_seed: int | None = None) -> Path:
    """Return the new nested run path, or the historical flat path when unseeded."""
    directory = Path(root).expanduser().resolve() / str(seed)
    return directory if random_seed is None else directory / str(random_seed)


def load_config(default_config: str = DEFAULT_CONFIG) -> DictConfig:
    """
    Parse the CLI and return the merged config.

    Usage: ``python train.py [--config PATH] [--resume] [--print-config] [key=value ...]``
    Overrides use OmegaConf dotlist syntax, e.g. ``epochs=5 aux_weights.bias=0.2``.
    A saved run config (``<save_dir>/<seed>/<random_seed>/config.yaml``) can be
    passed straight back in via ``--config`` to reproduce that run.

    Args:
        default_config: base YAML used when ``--config`` is not given.

    Returns:
        The base config merged with any CLI overrides.
    """
    parser = argparse.ArgumentParser(description="Load a YAML config with CLI overrides.")
    parser.add_argument(
        "--config", default=default_config,
        help="path to the base YAML config (default: " + default_config + ")",
    )
    parser.add_argument(
        "--print-config", action="store_true",
        help="print the resolved config and exit",
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='resume from the latest checkpoint for the selected run',
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="OmegaConf overrides, e.g. seed=run1 epochs=5 aux_weights.bias=0.2",
    )
    args = parser.parse_args()

    cfg = load_config_values(args.config, args.overrides)
    if args.resume:
        cfg.resume = True

    if args.print_config:
        print(OmegaConf.to_yaml(cfg))
        raise SystemExit(0)

    return cfg


def save_config(cfg: DictConfig, path: str) -> None:
    """Write the exact config used for a run to a single reusable YAML file."""
    OmegaConf.save(cfg, path, resolve=True)


def parse_evaluation_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse single-run or experiment-directory evaluation arguments."""
    parser = argparse.ArgumentParser(description="Evaluate saved training runs.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--config', help='Path to the saved run config.yaml')
    source.add_argument('--seed', help='Run name under the model directory')
    source.add_argument(
        '--experiment-dir',
        help='Experiment directory containing random-seed run directories',
    )
    parser.add_argument(
        '--random-seed', type=int, default=None,
        help='Random-seed directory within the selected run',
    )
    parser.add_argument(
        '--model-dir', default=None,
        help='Model directory (default: save_dir from configs/base.yaml)',
    )
    parser.add_argument(
        '--num-cores', type=positive_integer, default=1,
        help='Maximum simultaneous evaluations for --experiment-dir (default: 1)',
    )
    args = parser.parse_args(argv)
    if args.experiment_dir:
        if args.model_dir is not None or args.random_seed is not None:
            parser.error('--model-dir and --random-seed cannot be used with --experiment-dir')
        args.experiment_dir = os.path.abspath(os.path.expanduser(args.experiment_dir))
        return args
    if args.num_cores != 1:
        parser.error('--num-cores can only be used with --experiment-dir')
    if args.config:
        if args.model_dir is not None or args.random_seed is not None:
            parser.error('--model-dir and --random-seed can only be used with --seed')
        args.config = os.path.abspath(os.path.expanduser(args.config))
        return args
    if args.seed in ('.', '..') or os.path.basename(args.seed) != args.seed:
        parser.error('--seed must be a run directory name')
    if args.random_seed is not None and args.random_seed < 0:
        parser.error('--random-seed must be an integer greater than or equal to 0')
    model_dir = args.model_dir or OmegaConf.load(DEFAULT_CONFIG).save_dir
    args.config = str(run_directory(model_dir, args.seed, args.random_seed) / 'config.yaml')
    return args


def evaluation_config_path(argv: list[str] | None = None) -> str:
    """Locate one saved run YAML without accepting experiment-directory mode."""
    args = parse_evaluation_args(argv)
    if args.experiment_dir:
        raise ValueError('--experiment-dir does not identify a single run config.')
    return args.config
