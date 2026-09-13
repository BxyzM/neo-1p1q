"""
Evaluate one saved run or aggregate every random-seed run in an experiment.

Author: Aritra Bal (ETP)
Date: 2026-09-09
"""
from collections import deque
import glob
import os
import pathlib
import signal
import subprocess
import sys
import time
from typing import Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from loguru import logger
import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import roc_curve, roc_auc_score

import case_reader as cr
import helpers.utils as ut
import quantum.losses as loss
from helpers.config import SINGLE_THREAD_ENV, parse_evaluation_args, run_directory
from helpers.trained_run import CIRCUIT_FILES, load_trained_run


def _plot_score_distribution(scores: np.ndarray, labels: np.ndarray, path: str, xlabel: str) -> None:
    """Save a signal-vs-background classifier score histogram to path."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.hist(scores[labels == 1], bins=30, alpha=0.5, density=True, label='signal')
    ax.hist(scores[labels == 0], bins=30, alpha=0.5, density=True, label='background')
    ax.set_xlabel(xlabel, size=16)
    ax.set_ylabel('Density', size=16)
    ax.legend(prop={'size': 14})
    fig.savefig(path)
    plt.close(fig)


def main(config_path: str) -> dict[str, object]:
    """Verify a saved run, evaluate its final weights, and report test ROC/AUC."""
    config_path = str(pathlib.Path(config_path).expanduser().resolve())
    cfg, VQC = load_trained_run(config_path)
    random_seed = cfg.get('random_seed')
    save_dir = os.path.dirname(config_path)
    dump_dir = str(run_directory(cfg.dump, cfg.seed, random_seed))
    plot_dir = os.path.join(save_dir, 'plots')
    pathlib.Path(dump_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)

    logger.add(os.path.join(save_dir, 'eval.log'), rotation='10 MB', level='DEBUG')

    logger.info(f"Loaded verified final weights from {os.path.join(save_dir, 'trained_model.pickle')}")

    cost_fn = loss.VQC_cost

    test_split = 'flat_test' if cfg.flat else 'test'
    _class_files = lambda sample: sorted(glob.glob(os.path.join(cfg.data_dir, test_split, sample, '*.h5')))
    test_sig, test_bg = _class_files(cfg.signal), _class_files(cfg.background)
    if not (test_sig and test_bg):
        raise FileNotFoundError(
            f"Missing JetClass files under {cfg.data_dir} for signal='{cfg.signal}', "
            f"background='{cfg.background}' (split '{test_split}')"
        )
    logger.info(f"Test set: {cfg.n_signal_test} '{cfg.signal}' + {cfg.n_background_test} '{cfg.background}' jets from {test_split}/")

    required_particles = len(VQC.auto_wires)
    num_particles = getattr(cfg, 'num_particles', required_particles)
    if num_particles < required_particles:
        raise ValueError(
            f"The {cfg.num_layers}-layer circuit requires at least "
            f"{required_particles} particles, but num_particles={num_particles}"
        )

    test_loader = cr.OneP1QDataLoader(
        signal_filelist=test_sig, background_filelist=test_bg,
        n_signal=cfg.n_signal_test, n_background=cfg.n_background_test,
        input_shape=(num_particles, 3),
        train=False,
        normalize_pt=cfg.norm_pt,
        logger=logger,
        seed=random_seed if random_seed is not None else 0,
        batch_size=cfg.batch_size,
    )

    costs, scores, labels = VQC.run_inference(test_loader, loss_fn=cost_fn, loss_type=cfg.loss)

    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    logger.info(f"Test AUC: {auc:.4f}")

    results = {'costs': costs, 'scores': scores, 'labels': labels, 'auc': auc}
    ut.Pickle(results, 'test_results', path=dump_dir)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
    ax.set_xlabel('False Positive Rate', size=16)
    ax.set_ylabel('True Positive Rate', size=16)
    ax.legend(prop={'size': 14})
    fig.savefig(os.path.join(plot_dir, 'roc_curve.png'))
    plt.close(fig)
    logger.info(f"Saved ROC curve to {os.path.join(plot_dir, 'roc_curve.png')}")

    score_dist_path = os.path.join(plot_dir, 'score_distribution.png')
    is_logits = cfg.loss == 'BCE'
    probabilities = ut.sigmoid(scores) if is_logits else scores
    xlabel = 'Predicted probability' if is_logits else 'Classifier score'
    _plot_score_distribution(probabilities, labels, score_dist_path, xlabel)
    logger.info(f"Saved score distribution to {score_dist_path}")
    return results


def discover_evaluation_runs(
    experiment_dir: pathlib.Path,
) -> tuple[int, list[pathlib.Path], dict[str, str], list[str]]:
    """Find complete numeric random-seed run directories under an experiment."""
    seed_directories = []
    notices = []
    for path in sorted(experiment_dir.iterdir(), key=lambda item: item.name):
        if not path.is_dir():
            continue
        if not path.name.isdecimal():
            notices.append(f'Ignoring non-random-seed directory: {path}')
            continue
        seed_directories.append(path)
    seed_directories.sort(key=lambda path: (int(path.name), path.name))

    runnable = []
    failures = {}
    required = ['config.yaml', 'trained_model.pickle'] + [
        str(pathlib.Path('circuits') / name) for name in CIRCUIT_FILES
    ]
    for run_dir in seed_directories:
        missing = [name for name in required if not (run_dir / name).is_file()]
        if missing:
            failures[run_dir.name] = 'missing ' + ', '.join(missing)
        else:
            runnable.append(run_dir / 'config.yaml')
    return len(seed_directories), runnable, failures, notices


def _start_evaluation(config_path: pathlib.Path, gpu_id: int) -> subprocess.Popen:
    """Start one single-threaded evaluation subprocess.

    Mirrors run_experiments.py's _start_training: a jax-backend run (read from
    its own saved config.yaml) gets XLA_PYTHON_CLIENT_PREALLOCATE=false (JAX's
    default eager GPU memory preallocation can OOM a second concurrent jax
    subprocess on a shared GPU) and CUDA_VISIBLE_DEVICES=gpu_id (confines this
    process to exactly the one GPU it was assigned). Both must be set before
    `import jax` happens in that process, i.e. in its environment before Popen
    starts it. Autograd-backend runs are unaffected.
    """
    environment = os.environ.copy()
    environment.update(SINGLE_THREAD_ENV)
    backend = OmegaConf.load(config_path).get('backend')
    suffix = ''
    if backend == 'jax':
        environment['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        environment['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        suffix = f' on GPU {gpu_id}'
    command = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        '--config',
        str(config_path),
    ]
    print(f'Launching evaluation for random_seed={config_path.parent.name}{suffix}')
    return subprocess.Popen(command, env=environment, start_new_session=True)


def evaluate_configs(
    config_paths: list[pathlib.Path],
    num_cores: int,
    gpu_ids: Sequence[int] = (0,),
) -> tuple[list[pathlib.Path], dict[str, str], bool]:
    """Evaluate configs with bounded parallelism while retaining every failure."""
    pending = deque(config_paths)
    active: dict[subprocess.Popen, pathlib.Path] = {}
    successful = []
    failures = {}
    launched = 0

    try:
        while pending or active:
            while pending and len(active) < num_cores:
                config_path = pending.popleft()
                gpu_id = gpu_ids[launched % len(gpu_ids)]
                try:
                    process = _start_evaluation(config_path, gpu_id)
                except OSError as error:
                    failures[config_path.parent.name] = f'could not launch: {error}'
                    continue
                active[process] = config_path
                launched += 1

            completed = []
            while active and not completed:
                for process in active:
                    return_code = process.poll()
                    if return_code is not None:
                        completed.append((process, return_code))
                if not completed:
                    time.sleep(0.1)

            for process, return_code in completed:
                config_path = active.pop(process)
                random_seed = config_path.parent.name
                if return_code == 0:
                    successful.append(config_path)
                    print(f'Completed evaluation for random_seed={random_seed}')
                else:
                    failures[random_seed] = f'evaluation exited with code {return_code}'
    except KeyboardInterrupt:
        for process in active:
            try:
                process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
        for process in active:
            process.wait()
        return successful, failures, True

    return successful, failures, False


def _load_result(config_path: pathlib.Path) -> dict[str, object]:
    """Load and validate the newly written result for one successful evaluation."""
    cfg = OmegaConf.load(config_path)
    result_path = run_directory(
        cfg.dump, cfg.seed, cfg.get('random_seed')
    ) / 'test_results.pickle'
    results = ut.Unpickle(result_path)
    if not isinstance(results, dict) or not {'scores', 'labels', 'auc'} <= results.keys():
        raise ValueError(f'Invalid evaluation result: {result_path}')
    scores = np.asarray(results['scores']).reshape(-1)
    labels = np.asarray(results['labels']).reshape(-1)
    if not len(labels) or len(scores) != len(labels):
        raise ValueError(f'Invalid score and label lengths in {result_path}')
    auc = float(results['auc'])
    if not np.isfinite(auc):
        raise ValueError(f'Nonfinite AUC in {result_path}')
    return {
        'scores': scores,
        'labels': labels,
        'auc': auc,
    }


def summarize_evaluations(
    experiment_dir: pathlib.Path,
    results: list[dict[str, object]],
    total_runs: int,
) -> str:
    """Create the aggregate ROC plot and return the printable summary report."""
    successful_runs = len(results)
    failed_runs = total_runs - successful_runs
    lines = [
        'Evaluation summary',
        f'Runs in total: {total_runs}',
        f'Runs evaluated successfully: {successful_runs}',
        f'Runs skipped or failed: {failed_runs}',
    ]
    if not results:
        lines.extend(['Jets evaluated: 0', 'AUC: unavailable'])
        return '\n'.join(lines)

    jet_counts = [len(result['labels']) for result in results]
    if len(set(jet_counts)) == 1:
        lines.append(f'Jets evaluated per run: {jet_counts[0]}')
    else:
        lines.append(
            f'Jets evaluated per run: {min(jet_counts)}-{max(jet_counts)}'
        )
    lines.append(f'Total jet evaluations: {sum(jet_counts)}')

    aucs = np.asarray([result['auc'] for result in results], dtype=float)
    mean_auc = float(np.mean(aucs))
    auc_std = float(np.std(aucs))
    lines.append(
        f'AUC: {mean_auc:.4f} +/- {auc_std:.4f} '
        '(mean +/- standard deviation)'
    )

    fpr_grid = np.linspace(0.0, 1.0, 1001)
    interpolated_tprs = []
    for result in results:
        fpr, tpr, _ = roc_curve(result['labels'], result['scores'])
        interpolated_tprs.append(np.interp(fpr_grid, fpr, tpr))
    interpolated_tprs = np.asarray(interpolated_tprs)
    mean_tpr = np.mean(interpolated_tprs, axis=0)
    tpr_std = np.std(interpolated_tprs, axis=0)
    mean_tpr[0], mean_tpr[-1] = 0.0, 1.0

    plot_path = experiment_dir / 'roc_curve_summary.png'
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(
        fpr_grid, mean_tpr, linewidth=2,
        label=f'Mean AUC = {mean_auc:.4f} +/- {auc_std:.4f}',
    )
    ax.fill_between(
        fpr_grid,
        np.clip(mean_tpr - tpr_std, 0.0, 1.0),
        np.clip(mean_tpr + tpr_std, 0.0, 1.0),
        alpha=0.25,
        label='+/- 1 standard deviation',
    )
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
    ax.set_xlabel('False Positive Rate', size=16)
    ax.set_ylabel('True Positive Rate', size=16)
    ax.legend(prop={'size': 12})
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    lines.append(f'Aggregate ROC plot: {plot_path}')
    return '\n'.join(lines)


def evaluate_experiment(experiment_dir: str, num_cores: int, gpu_ids: Sequence[int] = (0,)) -> int:
    """Evaluate and summarize all random-seed runs under one experiment."""
    root = pathlib.Path(experiment_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f'Experiment directory does not exist: {root}')
    log_path = root / 'evaluation_summary.log'
    log_path.write_text('', encoding='utf-8')

    total_runs, config_paths, failures, notices = discover_evaluation_runs(root)
    for notice in notices:
        print(notice, file=sys.stderr)
    for random_seed, reason in failures.items():
        print(f'Skipping random_seed={random_seed}: {reason}', file=sys.stderr)

    successful, runtime_failures, interrupted = evaluate_configs(
        config_paths, num_cores, gpu_ids
    )
    failures.update(runtime_failures)
    if interrupted:
        message = 'Evaluation interrupted; active subprocesses were stopped.'
        log_path.write_text(message + '\n', encoding='utf-8')
        print(message, file=sys.stderr)
        return 130

    results = []
    for config_path in successful:
        try:
            results.append(_load_result(config_path))
        except (OSError, ValueError, TypeError) as error:
            failures[config_path.parent.name] = str(error)

    messages = notices + [
        f'random_seed={random_seed}: {reason}'
        for random_seed, reason in sorted(failures.items(), key=lambda item: int(item[0]))
    ]
    report = summarize_evaluations(root, results, total_runs)
    log_contents = '\n'.join(messages + ['', report]).lstrip()
    log_path.write_text(log_contents + '\n', encoding='utf-8')
    print(report)
    return 0 if results else 1


if __name__ == "__main__":
    arguments = parse_evaluation_args()
    if arguments.experiment_dir:
        raise SystemExit(
            evaluate_experiment(arguments.experiment_dir, arguments.num_cores, arguments.gpu_id)
        )
    main(arguments.config)
