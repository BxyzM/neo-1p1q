"""Training and persistence for the controlled depth/readout experiment.

The original train.py path remains separate: this study follows polarization's
cached-angle, AdamW, fixed-epoch protocol with Torch or JAX numerical execution.
"""

from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F

from helpers.depth_metrics import classification_metrics, write_csv
from helpers.depth_config import learning_rate, run_path, study_run_config, validate_study
from helpers.feature_cache import prepare_splits
from quantum.circuits.reupload import ReuploadingClassifier


SOURCE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "run_depth_readout.py", "helpers/depth_study.py", "helpers/depth_metrics.py",
    "helpers/depth_config.py", "helpers/depth_jax.py",
    "helpers/feature_cache.py", "quantum/circuits/base.py", "quantum/circuits/vqc.py",
    "quantum/circuits/reupload.py", "helpers/utils.py",
)


def implementation_hashes():
    return {name: hashlib.sha256((SOURCE_ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}




def atomic_json(path, payload):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}-", delete=False) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def atomic_checkpoint(path, payload):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}-", delete=False) as stream:
        torch.save(payload, stream)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def runtime_metadata(backend):
    packages = {}
    names = ["PennyLane", "torch", "numpy", "h5py", "scikit-learn", "omegaconf"]
    if not isinstance(backend, TorchBackend):
        names.extend(("jax", "jaxlib", "optax"))
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=SOURCE_ROOT, capture_output=True, text=True, check=False).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=SOURCE_ROOT, capture_output=True, text=True, check=False).stdout
    return {
        "python": platform.python_version(), "packages": packages,
        "platform": platform.platform(), "hostname": platform.node(),
        "cpu": platform.processor(), "cpu_count": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "XLA_FLAGS")},
        "git_revision": revision, "git_status": status,
        **backend.runtime_info(),
    }


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def batch_tensors(split, indices, dtype, device):
    # Copy only the selected batch from read-only mmap storage. No per-event
    # Python object collation and no full split transfer to the accelerator.
    angles = torch.from_numpy(np.array(split.angles[indices], copy=True)).to(device=device, dtype=dtype)
    labels = torch.from_numpy(np.array(split.labels[indices], copy=True)).to(device=device, dtype=dtype)
    return angles, labels


def predict(model, split, batch_size, device):
    synchronize(device)
    start = time.perf_counter()
    scores = np.empty(len(split.labels), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(scores), batch_size):
            indices = slice(offset, offset + batch_size)
            angles, _ = batch_tensors(split, indices, model.rotations.dtype, device)
            scores[indices] = model(angles).detach().cpu().numpy()
    synchronize(device)
    inference_seconds = time.perf_counter() - start
    metric_start = time.perf_counter()
    metrics = classification_metrics(split.labels, scores)
    metrics["inference_seconds"] = inference_seconds
    metrics["metric_seconds"] = time.perf_counter() - metric_start
    return metrics, scores


def train_epoch(model, split, optimizer, training, seed, device):
    model.train()
    synchronize(device)
    start = time.perf_counter()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(split.labels), generator=generator).numpy()
    total_loss, updates = 0.0, 0
    diagnostics = {"rotation_grad_norm": 0.0, "deformation_grad_abs": 0.0, "bias_grad_abs": 0.0, "readout_grad_norm": 0.0}
    for offset in range(0, len(order), training["batch_size"]):
        indices = order[offset:offset + training["batch_size"]]
        optimizer.zero_grad(set_to_none=True)
        for part in range(0, len(indices), training["microbatch_size"]):
            selected = indices[part:part + training["microbatch_size"]]
            angles, labels = batch_tensors(split, selected, model.rotations.dtype, device)
            logits = model(angles)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite training BCE")
            (loss * (len(selected) / len(indices))).backward()
            total_loss += float(loss.detach()) * len(selected)
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f"Missing/nonfinite gradient for {name}")
        diagnostics["rotation_grad_norm"] += float(model.rotations.grad.norm())
        diagnostics["deformation_grad_abs"] += float(model.deformation.grad.abs())
        diagnostics["bias_grad_abs"] += float(model.bias.grad.abs())
        if model.readout_weights is not None:
            diagnostics["readout_grad_norm"] += float(model.readout_weights.grad.norm())
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise FloatingPointError("Optimizer produced nonfinite parameters")
        updates += 1
    synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "train_online_bce": total_loss / len(order), "train_seconds": elapsed,
        "train_events_per_second": len(order) / elapsed, "optimizer_updates": updates,
        **{name: value / updates for name, value in diagnostics.items()},
    }


class TorchBackend:
    """Torch numerical state behind the shared study lifecycle."""

    def __init__(self, config, depth, readout, seed):
        training = config["training"]
        self.device = torch.device(training["torch_device"])
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable; select training.torch_device=cpu explicitly")
            torch.cuda.set_device(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.model = ReuploadingClassifier(config["circuit"]["wires"], depth, readout, seed,
                                           config["circuit"]["input_angle_scale"], training["precision"]).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.optimizer_groups(training["weight_decay"]), lr=training["initial_lr"])
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self.compilation = {"jit_compile_seconds": 0.0}

    def prepare(self):
        pass

    def predict(self, split, batch_size):
        return predict(self.model, split, batch_size, self.device)

    def train_epoch(self, split, training, seed, lr):
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return train_epoch(self.model, split, self.optimizer, training, seed, self.device)

    def model_state(self):
        return {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()}

    def optimizer_state(self):
        return self.optimizer.state_dict()

    def load_model(self, state):
        self.model.load_state_dict(state)

    def load_optimizer(self, state):
        self.optimizer.load_state_dict(state)

    @property
    def probability_dtype(self):
        return self.model.probability_dtype

    def diagnostics(self):
        return {"deformation_factor": float(self.model.deformation_factor.detach()),
                "bias": float(self.model.bias.detach()),
                "readout_weight_norm": float(self.model.readout_weights.detach().norm()) if self.model.readout_weights is not None else 1.0}

    def memory(self):
        return {"peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(self.device) / 2 ** 20 if self.device.type == "cuda" else None,
                "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(self.device) / 2 ** 20 if self.device.type == "cuda" else None}

    def runtime_info(self):
        return {"interface": "torch", "torch_device": str(self.device), "torch_threads": torch.get_num_threads(),
                "cuda_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
                "gpu_total_memory_mib": torch.cuda.get_device_properties(self.device).total_memory / 2 ** 20 if self.device.type == "cuda" else None}


def train_run(config, dataset, depth, readout, seed, resume=False):
    validate_study(config)
    if dataset not in config["datasets"] or depth not in config["depths"] or readout not in config["readouts"] or seed not in config["seeds"]:
        raise ValueError("Requested run is outside the configured study")
    training = config["training"]
    torch.set_num_threads(training["threads"])
    if config["circuit"]["interface"] == "jax":
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        from helpers.depth_jax import JaxBackend
        backend = JaxBackend(config, depth, readout, seed)
    else:
        backend = TorchBackend(config, depth, readout, seed)
    path = run_path(config, dataset, depth, readout, seed)
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another worker already owns {path}") from error
        return _train_locked(config, dataset, depth, readout, seed, resume, path, backend)


def _train_locked(config, dataset, depth, readout, seed, resume, path, backend):
    run_config = study_run_config(config, dataset, depth, readout, seed)
    signature = implementation_hashes()
    expected = {"config": run_config, "implementation": signature}
    manifest = path / "configuration.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != expected:
            raise ValueError(f"Run configuration/source mismatch: {path}; choose another output_dir")
        if not resume:
            raise FileExistsError(f"Run exists: {path}; pass --resume")
    else:
        for name in SOURCE_FILES:
            destination = path / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SOURCE_ROOT / name, destination)
        atomic_json(manifest, expected)
    for name, digest in signature.items():
        if hashlib.sha256((path / "source" / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Run source snapshot changed: {name}")

    loading_start = time.perf_counter()
    splits, data_metadata = prepare_splits(config["output_dir"], config["datasets"][dataset], seed, config["circuit"]["wires"], config["loader"]["block_rows"])
    data_seconds = time.perf_counter() - loading_start
    data_file = path / "data.json"
    if data_file.exists() and json.loads(data_file.read_text()) != data_metadata:
        raise ValueError("Run data identity changed")
    atomic_json(data_file, data_metadata)
    runtime = runtime_metadata(backend)
    if (path / "results.json").exists():
        completed = json.loads((path / "results.json").read_text())
        if completed.get("status") != "complete" or any(completed.get(key) != value for key, value in expected.items()) or completed.get("data") != data_metadata:
            raise ValueError("Completed result provenance mismatch")
        if completed["runtime"]["packages"] != runtime["packages"] or completed["runtime"]["python"] != runtime["python"]:
            raise ValueError("Completed run environment differs from the saved dependency versions")
        print(f"Already complete: {path}", flush=True)
        return completed

    training = config["training"]
    checkpoint_path = path / "last_checkpoint.pt"
    history, best_epoch, best_auc = [], 0, float("-inf")
    best_state = None
    initial = None
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if state["identity"] != expected or state["data"] != data_metadata:
            raise ValueError("Checkpoint configuration, source or data mismatch")
        if state["runtime"]["packages"] != runtime["packages"] or state["runtime"]["python"] != runtime["python"]:
            raise ValueError("Resume environment differs from the saved dependency versions")
        backend.load_model(state["model"])
        backend.load_optimizer(state["optimizer"])
        history, initial = state["history"], state["initial"]
        best_epoch, best_auc, best_state = state["best_epoch"], state["best_auc"], state["best_model"]
        if [row["epoch"] for row in history] != list(range(1, len(history) + 1)) or len(history) > training["epochs"]:
            raise ValueError("Invalid checkpoint epoch/history sequence")
    backend.prepare()
    if initial is None:
        initial_train, _ = backend.predict(splits["train"], training["eval_batch_size"])
        initial_valid, _ = backend.predict(splits["valid"], training["eval_batch_size"])
        initial = {"train": initial_train, "valid": initial_valid}
        atomic_json(path / "initial_metrics.json", initial)

    session = {"started_utc": datetime.now(timezone.utc).isoformat(), "runtime": runtime, "data_loading_seconds": data_seconds, "start_epoch": len(history) + 1, **backend.compilation}
    sessions_file = path / "sessions.json"
    sessions = json.loads(sessions_file.read_text()) if sessions_file.exists() else []
    sessions.append(session)
    atomic_json(sessions_file, sessions)
    for epoch in range(len(history), training["epochs"]):
        epoch_start = time.perf_counter()
        lr = learning_rate(epoch, training)
        train_metrics = backend.train_epoch(splits["train"], training, seed + epoch, lr)
        valid_metrics, _ = backend.predict(splits["valid"], training["eval_batch_size"])
        row = {
            "epoch": epoch + 1, "learning_rate": lr, **train_metrics,
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
            "epoch_seconds": time.perf_counter() - epoch_start,
            **backend.diagnostics(),
            "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            **backend.memory(),
        }
        history.append(row)
        if valid_metrics["auc"] > best_auc:
            best_auc, best_epoch = valid_metrics["auc"], epoch + 1
            best_state = backend.model_state()
        atomic_checkpoint(checkpoint_path, {
            "identity": expected, "data": data_metadata, "runtime": runtime,
            "model": backend.model_state(), "optimizer": backend.optimizer_state(),
            "history": history, "initial": initial,
            "best_model": best_state, "best_epoch": best_epoch, "best_auc": best_auc,
        })
        atomic_json(path / "history.json", history)
        write_csv(path / "history.csv", history)
        print(f"{dataset} L={depth} {readout} seed={seed} epoch={epoch + 1}/{training['epochs']} train_BCE={row['train_online_bce']:.6f} valid_BCE={row['valid_bce']:.6f} AUC={row['valid_auc']:.6f} train_s={row['train_seconds']:.2f} epoch_s={row['epoch_seconds']:.2f}", flush=True)

    # Final fixed-weights train BCE makes the before/after reduction comparable;
    # online epoch loss mixes predictions made at different optimizer states.
    final_train, _ = backend.predict(splits["train"], training["eval_batch_size"])
    final_state = backend.model_state()
    backend.load_model(best_state)
    best_valid, scores = backend.predict(splits["valid"], training["eval_batch_size"])
    np.savez_compressed(path / "validation_predictions.npz", logits=scores, labels=splits["valid"].labels, source_rows=splits["valid"].rows, event_fingerprints=splits["valid"].fingerprints)
    model_artifact = {"identity": expected, "data": data_metadata, "best_epoch": best_epoch, "best_model": best_state, "final_model": final_state}
    atomic_checkpoint(path / "models.pt", model_artifact)
    parameter_count = backend.parameter_count
    final_valid_bce = history[-1]["valid_bce"]
    train_drop = initial["train"]["bce"] - final_train["bce"]
    valid_drop = initial["valid"]["bce"] - final_valid_bce
    peak_memory = {}
    for name, current in backend.memory().items():
        values = [value for value in [current, *(row.get(name) for row in history)] if value is not None]
        peak_memory[name] = max(values) if values else None
    summary = {
        "initial_train_bce": initial["train"]["bce"], "final_train_bce": final_train["bce"],
        "train_bce_drop": train_drop, "train_bce_drop_percent": 100 * train_drop / initial["train"]["bce"],
        "initial_valid_bce": initial["valid"]["bce"], "final_valid_bce": final_valid_bce,
        "valid_bce_drop": valid_drop, "valid_bce_drop_percent": 100 * valid_drop / initial["valid"]["bce"],
        "minimum_valid_bce": min(row["valid_bce"] for row in history),
        "minimum_valid_bce_epoch": min(history, key=lambda row: row["valid_bce"])["epoch"],
        "best_auc": best_auc, "best_auc_epoch": best_epoch, "bce_at_best_auc": best_valid["bce"],
        "final_auc": history[-1]["valid_auc"], "generalization_bce_gap": final_valid_bce - final_train["bce"],
        "mean_train_seconds": float(np.mean([row["train_seconds"] for row in history])),
        "mean_epoch_seconds": float(np.mean([row["epoch_seconds"] for row in history])),
        "total_train_seconds": float(sum(row["train_seconds"] for row in history)),
        "total_epoch_seconds": float(sum(row["epoch_seconds"] for row in history)),
        "mean_valid_inference_seconds": float(np.mean([row["valid_inference_seconds"] for row in history])),
        "mean_train_events_per_second": float(np.mean([row["train_events_per_second"] for row in history])),
        "batch_size": training["batch_size"], "microbatch_size": training["microbatch_size"], "eval_batch_size": training["eval_batch_size"],
        "train_events": len(splits["train"].labels), "valid_events": len(splits["valid"].labels),
        "parameters": parameter_count, "quantum_parameters": 3 * config["circuit"]["wires"] * depth + 1,
        "readout_parameters": parameter_count - 3 * config["circuit"]["wires"] * depth - 1,
        "known_noninfluential_final_rotation_parameters": 3 * (config["circuit"]["wires"] - 1) if readout == "z0" else 0,
        "cnot_gates": config["circuit"]["wires"] * depth,
        "peak_process_rss_mib": max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, *(row["peak_process_rss_mib"] for row in history)),
        **peak_memory,
        "total_jit_compile_seconds": sum(session.get("jit_compile_seconds", 0.0) for session in sessions),
        **{f"best_valid_{name}": value for name, value in best_valid.items()},
    }
    result = {
        "status": "complete", **expected, "data": data_metadata,
        "runtime": runtime, "probability_dtype": backend.probability_dtype,
        "initial": initial, "final_train": final_train, "best_valid": best_valid,
        "summary": summary, "history": history,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": {
            "scope": "Top/QCD validation scan; test split was not read.",
            "timing": "epoch_seconds=train+validation inference+metrics; excludes checkpoint IO and JIT compilation. Initial/final diagnostics, data preparation and per-session compilation are separate.",
            "seeds": "Each seed controls initialization, per-epoch shuffle, and the selected subsets; variation is not initialization-only.",
            "selection": "Best model selected by validation ROC AUC among trained epochs; minimum validation BCE is reported separately.",
            "identity": "JetGame has audited content fingerprints. JetClass identifiers are split/row hashes and cannot rule out physical duplicates.",
            "state": "PennyLane may promote internal precision; probability_dtype is the observed output precision before readout casting.",
        },
    }
    atomic_json(path / "results.json", result)
    return result
