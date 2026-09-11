"""Configuration contracts and deterministic job assignment for the depth study."""

import math
from pathlib import Path
import re

READOUTS = ("z0", "all_z", "all_z_zz")


def validate_study(config):
    required = {"output_dir", "depths", "readouts", "seeds", "datasets", "circuit", "training", "loader"}
    if set(config) != required:
        raise ValueError(f"Study keys must be {sorted(required)}; got {sorted(config)}")
    for name in ("depths", "seeds", "readouts"):
        values = config[name]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"{name} must be a nonempty list without duplicates")
    if any(type(x) is not int or not 1 <= x <= 128 for x in config["depths"]):
        raise ValueError("Depths must be integers in [1, 128]")
    if any(type(x) is not int or not 0 <= x < 2 ** 31 for x in config["seeds"]):
        raise ValueError("Seeds must be nonnegative int32 integers")
    if any(x not in READOUTS for x in config["readouts"]):
        raise ValueError(f"Readouts must be drawn from {READOUTS}")
    circuit = config["circuit"]
    expected_circuit = {"device_name": "default.qubit", "diff_method": "backprop", "shots": None}
    if set(circuit) != {*expected_circuit, "interface", "wires", "input_angle_scale"}:
        raise ValueError("Unexpected/missing circuit configuration fields")
    for key, value in expected_circuit.items():
        if circuit[key] != value:
            raise ValueError(f"This study requires {key}={value!r}")
    if circuit["interface"] not in ("torch", "jax"):
        raise ValueError("circuit.interface must be torch or jax")
    if type(circuit["wires"]) is not int or circuit["wires"] < 2:
        raise ValueError("wires must be an integer >= 2")
    if not math.isfinite(circuit["input_angle_scale"]) or circuit["input_angle_scale"] <= 0:
        raise ValueError("input_angle_scale must be positive and finite")
    training = config["training"]
    device_key = "jax_device" if circuit["interface"] == "jax" else "torch_device"
    required_training = {"epochs", "warmup_epochs", "initial_lr", "peak_lr", "final_lr", "weight_decay", "batch_size", "microbatch_size", "eval_batch_size", "precision", device_key, "threads"}
    if set(training) != required_training:
        raise ValueError("Unexpected/missing training configuration fields")
    for name in ("epochs", "warmup_epochs", "batch_size", "microbatch_size", "eval_batch_size", "threads"):
        if type(training[name]) is not int or training[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if training["warmup_epochs"] > training["epochs"]:
        raise ValueError("warmup_epochs cannot exceed epochs")
    if training["microbatch_size"] > training["batch_size"]:
        raise ValueError("microbatch_size cannot exceed the optimizer batch_size")
    for name in ("initial_lr", "peak_lr", "final_lr"):
        if not math.isfinite(training[name]) or training[name] <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not math.isfinite(training["weight_decay"]) or training["weight_decay"] < 0:
        raise ValueError("weight_decay must be nonnegative and finite")
    if training["precision"] not in ("float32", "float64"):
        raise ValueError("precision must be float32 or float64")
    pattern = r"cpu|gpu(?::\d+)?" if circuit["interface"] == "jax" else r"cpu|cuda(?::\d+)?"
    if not isinstance(training[device_key], str) or not re.fullmatch(pattern, training[device_key]):
        raise ValueError(f"Invalid {device_key}: {training[device_key]!r}")
    if not config["datasets"] or not set(config["datasets"]).issubset({"jetclass", "jetgame"}):
        raise ValueError("Select jetclass and/or jetgame datasets")
    for dataset in config["datasets"].values():
        if set(dataset) != {"cache", "max_train_events", "max_valid_events"}:
            raise ValueError("Unexpected/missing dataset configuration fields")
        for name in ("max_train_events", "max_valid_events"):
            value = dataset[name]
            if value is not None and (type(value) is not int or value < 4 or value % 2):
                raise ValueError(f"{name} must be null or an even integer >= 4")
    if set(config["loader"]) != {"block_rows"} or type(config["loader"]["block_rows"]) is not int or config["loader"]["block_rows"] < 1:
        raise ValueError("loader.block_rows must be a positive integer")


def study_run_config(config, dataset, depth, readout, seed):
    return {
        "dataset": dataset, "depth": depth, "readout": readout, "seed": seed,
        "data": config["datasets"][dataset], "circuit": config["circuit"],
        "training": config["training"], "loader": config["loader"],
        "task": "top_vs_qcd", "loss": "BCEWithLogits", "optimizer": "AdamW",
        "initialization": "legacy_zero_bias", "training_positive_class": "signal",
        "encoding": "same_particle_reuploading", "rotation_placement": "after_entangler",
        "selection_metric": "validation_auc", "validation_only": True,
    }


def run_path(config, dataset, depth, readout, seed):
    return Path(config["output_dir"]) / dataset / f"L{depth:03d}" / readout / str(seed)


def partition_jobs(config, dataset, count):
    """Deterministic greedy partition by depth cost; every job appears once."""
    if type(count) is not int or count < 1:
        raise ValueError("Shard count must be a positive integer")
    jobs = [(depth, readout, seed) for depth in config["depths"] for readout in config["readouts"] for seed in config["seeds"]]
    shards, costs = [[] for _ in range(count)], [0] * count
    for job in sorted(jobs, key=lambda item: (-item[0], item[1], item[2])):
        shard = min(range(count), key=lambda index: (costs[index], len(shards[index]), index))
        shards[shard].append(job)
        costs[shard] += job[0]
    return [sorted(shard, key=lambda item: (item[0], item[1], item[2])) for shard in shards]


def learning_rate(epoch, training):
    """Zero-based linear warmup/cosine schedule matching polarization."""
    warmup = training["warmup_epochs"]
    initial, peak, final = (training[key] for key in ("initial_lr", "peak_lr", "final_lr"))
    if epoch < warmup:
        fraction = 1.0 if warmup == 1 else epoch / (warmup - 1)
        return initial + (peak - initial) * fraction
    progress = (epoch - warmup + 1) / (training["epochs"] - warmup)
    return final + (peak - final) * 0.5 * (1 + math.cos(math.pi * progress))
