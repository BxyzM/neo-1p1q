"""Metrics and descriptive five-seed statistics for the reuploading study."""

import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def classification_metrics(labels, logits):
    labels = np.asarray(labels)
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != labels.shape or not np.isfinite(logits).all():
        raise ValueError("Expected one finite logit per event")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Metrics require both binary classes")
    # Stable sigmoid and BCE, including confidently wrong predictions.
    probability = np.exp(-np.logaddexp(0, -logits))
    prediction = logits >= 0
    signal = labels == 1
    background = ~signal
    metrics = {
        "bce": float(np.mean(np.logaddexp(0, logits) - labels * logits)),
        "auc": float(roc_auc_score(labels, logits)),
        "average_precision": float(average_precision_score(labels, logits)),
        "accuracy": float(np.mean(prediction == labels)),
        "balanced_accuracy": float((prediction[signal].mean() + (~prediction[background]).mean()) / 2),
        "brier": float(np.mean((probability - labels) ** 2)),
        "signal_logit_mean": float(logits[signal].mean()),
        "background_logit_mean": float(logits[background].mean()),
        "logit_std": float(logits.std()),
        "true_positive": int(np.sum(prediction & signal)),
        "false_positive": int(np.sum(prediction & background)),
        "true_negative": int(np.sum(~prediction & background)),
        "false_negative": int(np.sum(~prediction & signal)),
        "events": len(labels),
    }
    fpr, tpr, thresholds = roc_curve(labels, logits, drop_intermediate=False)
    for efficiency in (0.3, 0.5, 0.8):
        index = int(np.flatnonzero(tpr >= efficiency)[0])
        suffix = str(int(100 * efficiency))
        # Empirical >= threshold operating point, with no ROC interpolation.
        accepted = int(np.sum(logits[background] >= thresholds[index]))
        metrics[f"fpr_at_tpr{suffix}"] = float(fpr[index])
        metrics[f"achieved_tpr{suffix}"] = float(tpr[index])
        metrics[f"threshold_at_tpr{suffix}"] = float(thresholds[index])
        metrics[f"background_accepted_at_tpr{suffix}"] = accepted
        # Zero observed background is unresolved by this sample, not infinity.
        metrics[f"rejection_at_tpr{suffix}"] = float(1 / fpr[index]) if fpr[index] > 0 else None
    return metrics


def statistics(values):
    """Sample SD and a two-sided Student-t interval for the mean (n <= 5).

    This describes variation across runs; it is not an independent event-level
    generalization interval, a discovery significance, or a multiple-test fix.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    count = len(values)
    if not count:
        return {"n": 0, "mean": None, "sd": None, "sem": None, "ci95_low": None, "ci95_high": None}
    mean = float(values.mean())
    if count < 2:
        return {"n": count, "mean": mean, "sd": None, "sem": None, "ci95_low": None, "ci95_high": None}
    # scipy is already a required dependency of PennyLane and scikit-learn.
    from scipy.stats import t
    sd = float(values.std(ddof=1))
    sem = sd / math.sqrt(count)
    width = float(t.ppf(0.975, count - 1)) * sem
    return {"n": count, "mean": mean, "sd": sd, "sem": sem, "ci95_low": mean - width, "ci95_high": mean + width}


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_study(config, output_dir):
    """Summarize only completed, protocol-compatible runs; expose missing seeds."""
    from helpers.depth_config import study_run_config, run_path
    output_dir = Path(output_dir)
    runs, missing = [], []
    for dataset in config["datasets"]:
        for depth in config["depths"]:
            for readout in config["readouts"]:
                for seed in config["seeds"]:
                    expected = study_run_config(config, dataset, depth, readout, seed)
                    path = run_path(config, dataset, depth, readout, seed)
                    if not (path / "results.json").is_file():
                        missing.append({"dataset": dataset, "depth": depth, "readout": readout, "seed": seed})
                        continue
                    result = json.loads((path / "results.json").read_text())
                    if result["config"] != expected or result.get("status") != "complete":
                        raise ValueError(f"Incompatible completed run: {path}")
                    runs.append(result)
    # Prevent merging code changes under one table even if YAML stayed fixed.
    if len({json.dumps(run["implementation"], sort_keys=True) for run in runs}) > 1:
        raise ValueError("Cannot aggregate runs from different implementations")
    if len({json.dumps((run["runtime"]["packages"], run["runtime"]["python"]), sort_keys=True) for run in runs}) > 1:
        raise ValueError("Cannot aggregate runs from different dependency versions")
    for dataset in config["datasets"]:
        identities = {
            json.dumps(run["data"]["identity"]["cache"], sort_keys=True)
            for run in runs if run["config"]["dataset"] == dataset
        }
        if len(identities) > 1:
            raise ValueError(f"Cannot aggregate different {dataset} source caches")
    rows = []
    lookup = {}
    for result in runs:
        cfg = result["config"]
        row = {key: cfg[key] for key in ("dataset", "depth", "readout", "seed")}
        row.update(result["summary"])
        rows.append(row)
        lookup[(cfg["dataset"], cfg["depth"], cfg["readout"], cfg["seed"])] = result
    grouped, paired = [], []
    for dataset in config["datasets"]:
        for depth in config["depths"]:
            for readout in config["readouts"]:
                selected = [row for row in rows if (row["dataset"], row["depth"], row["readout"]) == (dataset, depth, readout)]
                group = {"dataset": dataset, "depth": depth, "readout": readout, "completed_seeds": len(selected), "expected_seeds": len(config["seeds"])}
                keys = selected[0].keys() if selected else ()
                for metric in keys:
                    if metric in ("dataset", "depth", "readout", "seed"):
                        continue
                    values = [row[metric] for row in selected if isinstance(row[metric], (int, float))]
                    for statistic, value in statistics(values).items():
                        group[f"{metric}_{statistic}"] = value
                grouped.append(group)
                if readout == "z0":
                    continue
                for metric in ("best_auc", "final_valid_bce", "minimum_valid_bce", "valid_bce_drop", "mean_train_seconds"):
                    differences = []
                    for seed in config["seeds"]:
                        current = lookup.get((dataset, depth, readout, seed))
                        baseline = lookup.get((dataset, depth, "z0", seed))
                        if current is None or baseline is None:
                            continue
                        if current["data"]["splits"] != baseline["data"]["splits"]:
                            raise ValueError("Paired readouts used different data subsets")
                        differences.append(current["summary"][metric] - baseline["summary"][metric])
                    paired.append({"dataset": dataset, "depth": depth, "readout_minus_z0": readout, "metric": metric, **statistics(differences)})
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "runs.csv", rows)
    write_csv(output_dir / "depth_readout.csv", grouped)
    write_csv(output_dir / "paired_vs_z0.csv", paired)
    (output_dir / "coverage.json").write_text(json.dumps({
        "completed": len(rows), "expected": len(rows) + len(missing), "missing": missing,
        "scope": "Validation scan; no test data read. CI describes between-seed variability.",
    }, indent=2) + "\n")
    if rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for dataset in config["datasets"]:
            figure, axes = plt.subplots(1, 3, figsize=(14, 4))
            for readout in config["readouts"]:
                selected = [row for row in grouped if row["dataset"] == dataset and row["readout"] == readout and row["completed_seeds"]]
                for axis, metric, label in zip(axes, ("best_auc", "final_valid_bce", "mean_train_seconds"), ("Best validation ROC AUC", "Final validation BCE", "Training seconds / epoch")):
                    x = [row["depth"] for row in selected]
                    mean = [row[f"{metric}_mean"] for row in selected]
                    sd = [row[f"{metric}_sd"] or 0 for row in selected]
                    axis.errorbar(x, mean, yerr=sd, marker=".", label=readout, capsize=2)
                    axis.set(xlabel="Reuploading depth L", ylabel=label, xscale="log")
                    axis.grid(alpha=0.2)
            axes[0].legend()
            figure.suptitle(f"{dataset}: mean ± sample SD; see coverage.json for seed counts")
            figure.tight_layout()
            for suffix in ("png", "pdf"):
                figure.savefig(output_dir / f"{dataset}_depth_readout.{suffix}", dpi=180)
            plt.close(figure)
    return len(rows), len(missing)
