"""
Launch repeated training runs with randomly generated seeds and bounded parallelism.

Author: Aritra Bal (ETP)
Date: 2026-09-10
"""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Sequence

import numpy as np
from omegaconf import OmegaConf

from helpers.config import (
    DEFAULT_CONFIG,
    SINGLE_THREAD_ENV,
    gpu_id_list,
    load_config_values,
    positive_integer,
    run_directory,
    save_config,
    validate_training_config,
)

_RANDOM_SEED_HIGH = 2 ** 31  # Keeps generated seeds within the non-negative int32 range.


def _generate_random_seeds(count: int) -> list[int]:
    """Draw ``count`` distinct random seeds with NumPy's default generator."""
    rng = np.random.default_rng()
    seeds: list[int] = []
    while len(seeds) < count:
        candidate = int(rng.integers(0, _RANDOM_SEED_HIGH))
        if candidate not in seeds:
            seeds.append(candidate)
    return seeds


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse launcher settings separately from training overrides."""
    parser = argparse.ArgumentParser(
        description='Launch repeated training runs with randomly generated seeds.'
    )
    parser.add_argument(
        '--config', default=DEFAULT_CONFIG,
        help=f'path to the training YAML (default: {DEFAULT_CONFIG})',
    )
    parser.add_argument('--number-of-runs', type=positive_integer, default=1)
    parser.add_argument(
        '--num-processes', type=positive_integer, default=1,
        help='maximum simultaneous training processes (default: 1)',
    )
    parser.add_argument(
        '--gpu-id', type=gpu_id_list, default=[0],
        help="comma-separated GPU indices to use, e.g. '0' or '0,1' (backend='jax' only; "
             'default: 0). Concurrent processes are assigned these round-robin.',
    )
    parser.add_argument(
        'overrides', nargs='*',
        help='training overrides, e.g. seed=experiment epochs=5 (random_seed is generated per run)',
    )
    return parser.parse_args(argv)


def _start_training(
    config_path: str, random_seed: int, backend: str = 'autograd', gpu_id: int = 0,
) -> subprocess.Popen:
    """Start one single-threaded training subprocess.

    backend='jax' sets two things in the child's environment: (1)
    XLA_PYTHON_CLIENT_PREALLOCATE=false, since JAX's default eager GPU memory
    preallocation (~75% of the device on first touch) can make a second
    concurrent jax subprocess OOM on a shared GPU even with memory free; (2)
    CUDA_VISIBLE_DEVICES=gpu_id, confining this process to exactly the one GPU
    it was assigned -- without this, a jax process registers a small memory
    footprint on every visible GPU at startup even though it only computes on
    one. Both must be set before `import jax` happens in that process, i.e. in
    its environment before Popen starts it.
    """
    command = [
        sys.executable,
        str(Path(__file__).resolve().with_name('train.py')),
        '--config',
        config_path,
        f'random_seed={random_seed}',
    ]
    environment = os.environ.copy()
    environment.update(SINGLE_THREAD_ENV)
    if backend == 'jax':
        environment['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        environment['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    suffix = f' on GPU {gpu_id}' if backend == 'jax' else ''
    print(f'Launching random_seed={random_seed}{suffix}')
    return subprocess.Popen(command, env=environment, start_new_session=True)


def launch_runs(
    config_path: str, random_seeds: Sequence[int], num_processes: int,
    backend: str = 'autograd', gpu_ids: Sequence[int] = (0,),
) -> int:
    """Run at most ``num_processes`` training processes concurrently.

    For backend='jax', each launched process is assigned one GPU from
    ``gpu_ids`` in round-robin order -- ``num_processes`` and ``gpu_ids`` are
    independent: more processes than listed GPUs means some processes share a
    GPU's compute (safe, just slower per run); fewer means some listed GPUs go
    unused this round.
    """
    pending = iter(random_seeds)
    active: dict[subprocess.Popen, int] = {}
    failed = False
    exhausted = False
    launched = 0

    try:
        while active or not exhausted:
            while not failed and not exhausted and len(active) < num_processes:
                try:
                    random_seed = next(pending)
                except StopIteration:
                    exhausted = True
                    break
                gpu_id = gpu_ids[launched % len(gpu_ids)]
                try:
                    process = _start_training(config_path, random_seed, backend, gpu_id)
                except OSError as error:
                    failed = True
                    print(
                        f'Could not launch random_seed={random_seed}: {error}',
                        file=sys.stderr,
                    )
                    break
                active[process] = random_seed
                launched += 1

            if not active:
                break

            completed: list[tuple[subprocess.Popen, int]] = []
            while not completed:
                for process in active:
                    return_code = process.poll()
                    if return_code is not None:
                        completed.append((process, return_code))
                if not completed:
                    time.sleep(0.1)

            for process, return_code in completed:
                random_seed = active.pop(process)
                if return_code == 0:
                    print(f'Completed random_seed={random_seed}')
                else:
                    failed = True
                    print(
                        f'Training failed for random_seed={random_seed} '
                        f'with exit code {return_code}.',
                        file=sys.stderr,
                    )
    except KeyboardInterrupt:
        for process in active:
            if process.poll() is None:
                try:
                    process.send_signal(signal.SIGINT)
                except ProcessLookupError:
                    pass
        for process in active:
            process.wait()
        return 130

    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve one experiment and launch its requested independent runs."""
    args = _parse_args(argv)
    if any(override.split('=', 1)[0] == 'random_seed' for override in args.overrides):
        raise ValueError(
            'random_seed is generated per run by run_experiments.py and must not be overridden; '
            'set it directly with train.py instead.'
        )
    cfg = load_config_values(args.config, args.overrides)
    cfg = OmegaConf.create(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    validate_training_config(cfg)
    if cfg.get('resume', False):
        raise ValueError('run_experiments.py only launches fresh runs; resume runs individually.')

    backend = cfg.get('backend', 'autograd')
    if backend != 'jax' and args.gpu_id != [0]:
        print(
            "Warning: --gpu-id has no effect for backend='autograd' -- this pipeline has no "
            'GPU-accelerated autograd path.',
            file=sys.stderr,
        )
    if backend == 'jax' and args.num_processes > len(args.gpu_id):
        print(
            f'Note: --num-processes ({args.num_processes}) exceeds the number of GPUs listed in '
            f"--gpu-id ({len(args.gpu_id)}) -- --num-processes no longer maps to physical CPU "
            'cores under backend=\'jax\'; some processes will share a GPU\'s compute.',
            file=sys.stderr,
        )

    random_seeds = _generate_random_seeds(args.number_of_runs)

    legacy_config = run_directory(cfg.save_dir, cfg.seed) / 'config.yaml'
    if legacy_config.exists():
        raise FileExistsError(
            f'Legacy run already uses experiment directory {legacy_config.parent}.'
        )
    destinations = [run_directory(cfg.save_dir, cfg.seed, seed) for seed in random_seeds]
    existing = [str(path) for path in destinations if path.exists()]
    if existing:
        raise FileExistsError('Run directories already exist: ' + ', '.join(existing))

    with tempfile.TemporaryDirectory(prefix='neo1p1q-runs-') as directory:
        resolved_config = str(Path(directory) / 'config.yaml')
        save_config(cfg, resolved_config)
        return launch_runs(resolved_config, random_seeds, args.num_processes, backend, args.gpu_id)


if __name__ == '__main__':
    raise SystemExit(main())
