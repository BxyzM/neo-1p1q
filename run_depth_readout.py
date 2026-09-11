"""Run one of four disjoint study shards, or summarize the completed runs."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

from omegaconf import OmegaConf


def summarize(config, report_dir=None):
    from helpers.depth_metrics import summarize_study
    report = Path(report_dir or Path(config["output_dir"]) / "analysis")
    report.mkdir(parents=True, exist_ok=True)
    with (report / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        completed, missing = summarize_study(config, report)
    print(f"{completed} completed, {missing} missing; report: {report}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs/depth_readout.yaml"))
    parser.add_argument("--jax", action="store_true", help="select the JAX study overlay (separate output directory and GPU settings)")
    parser.add_argument("--dataset", choices=("jetclass", "jetgame"))
    parser.add_argument("--shard", default="0/1", help="zero-based shard/count; use 0/2 and 1/2 for each dataset")
    parser.add_argument("--resume", action="store_true", help="skip compatible completed runs and continue epoch checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="print assigned jobs without reading data or training")
    parser.add_argument("--summarize", action="store_true", help="write CSV statistics and PNG/PDF plots without training")
    parser.add_argument("--report-dir", help="summary output (default: <output_dir>/analysis)")
    parser.add_argument("--worker", nargs=3, metavar=("DEPTH", "READOUT", "SEED"), help=argparse.SUPPRESS)
    parser.add_argument("overrides", nargs="*", help="OmegaConf key=value overrides")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = OmegaConf.load(args.config)
    if args.jax:
        # A small overlay keeps the scientific protocol in one authoritative
        # base file and prevents accidental resumption of Torch artifacts.
        config = OmegaConf.merge(config, OmegaConf.load(Path(__file__).parent / "configs/depth_readout_jax.yaml"))
        config.training.pop("torch_device", None)
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))
    config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    config["output_dir"] = str(Path(config["output_dir"]).expanduser().resolve())
    for dataset in config["datasets"].values():
        dataset["cache"] = str(Path(dataset["cache"]).expanduser().resolve())
    # Set thread limits before importing NumPy, Torch, or PennyLane.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = str(config["training"]["threads"])
    from helpers.depth_config import partition_jobs, validate_study
    validate_study(config)
    if config["circuit"]["interface"] == "jax":
        # Set before any JAX import, also in fresh workers. Two processes share
        # each GPU; neither should eagerly reserve most of its memory.
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.summarize:
        summarize(config, args.report_dir)
        return 0
    if args.dataset is None:
        raise ValueError("--dataset is required for training and dry runs")
    if args.dataset not in config["datasets"]:
        raise ValueError(f"Dataset {args.dataset} is not configured")
    if args.worker:
        from helpers.depth_study import train_run
        depth, readout, seed = args.worker
        train_run(config, args.dataset, int(depth), readout, int(seed), resume=args.resume)
        return 0
    try:
        index, count = (int(part) for part in args.shard.split("/"))
    except ValueError as error:
        raise ValueError("--shard must be index/count, for example 0/2") from error
    if count < 1 or not 0 <= index < count:
        raise ValueError("Shard requires 0 <= index < count")
    jobs = partition_jobs(config, args.dataset, count)[index]
    print(f"{args.dataset} shard {index}/{count}: {len(jobs)} runs, {sum(job[0] for job in jobs)} depth units", flush=True)
    if args.dry_run:
        print(json.dumps([{"depth": depth, "readout": readout, "seed": seed} for depth, readout, seed in jobs], indent=2))
        return 0
    # Each job gets a fresh process, releasing the deep autograd graph and
    # allocator caches and making per-run peak RSS interpretable.
    with tempfile.TemporaryDirectory(prefix="depth-readout-") as temporary:
        frozen = Path(temporary) / "config.yaml"
        OmegaConf.save(OmegaConf.create(config), frozen)
        for depth, readout, seed in jobs:
            command = [sys.executable, str(Path(__file__).resolve()), "--config", str(frozen), "--dataset", args.dataset, "--worker", str(depth), readout, str(seed)]
            if args.resume:
                command.append("--resume")
            process = subprocess.Popen(command, start_new_session=True)
            try:
                returncode = process.wait()
            except KeyboardInterrupt:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
                process.wait()
                return 130
            if returncode:
                print(f"Failed: depth={depth}, readout={readout}, seed={seed}; resume after resolving the error", file=sys.stderr)
                return 1
    summarize(config, args.report_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
