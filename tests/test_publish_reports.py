"""
Verify static report publication, aggregation, and artifact failure handling.

Author: Aritra Bal (ETP)
Date: 2026-09-10
"""

import csv
import json
import shutil
import stat
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
from omegaconf import OmegaConf

import publish_reports
from helpers.trained_run import save_circuit_snapshot
from quantum.circuits.base import CircuitWeights


class TestReportPublication(unittest.TestCase):
    """The publisher must expose plot data without leaking local paths."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.models = self.root / "private" / "saved_models"
        self.results = self.root / "private" / "results"
        self.output = self.root / "public"
        self.site = self.root / "report_site"
        for relative_path in publish_reports.FRONTEND_FILES:
            path = self.site / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(relative_path), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_run(
        self,
        experiment: str,
        random_seed: int,
        validation_aucs: list[float],
        scores: list[float],
        labels: list[int],
        epoch_times: list[float] | None = None,
        compile_times: dict[str, float] | None = None,
    ) -> None:
        run_dir = self.models / experiment / str(random_seed)
        result_dir = self.results / experiment / str(random_seed)
        run_dir.mkdir(parents=True)
        result_dir.mkdir(parents=True)
        save_circuit_snapshot(str(run_dir))
        cfg = {
            "seed": experiment,
            "random_seed": random_seed,
            "signal": "TTBar_",
            "background": "ZJetsToNuNu",
            "n_signal": 10,
            "n_background": 10,
            "n_signal_val": 4,
            "n_background_val": 4,
            "n_signal_test": 2,
            "n_background_test": 2,
            "flat": True,
            "norm_pt": False,
            "data_dir": str(self.root / "private" / "JetClass"),
            "save_dir": str(self.models),
            "dump": str(self.results),
            "wires": 4,
            "num_layers": 1,
            "shots": -1,
            "backend": "autograd",
            "device_name": "lightning.qubit",
            "circuit_type": "normal",
            "operations_per_qubit": 3,
            "aux_weights": {"scale_factor": 1.0, "bias": 0.1},
            "batch_size": 5,
            "epochs": 25,
            "lr": 0.05,
            "improv": 0.01,
            "min_epochs": 10,
            "decay_rate": 0.5,
            "decay_patience": 3,
            "loss": "BCE",
        }
        OmegaConf.save(OmegaConf.create(cfg), run_dir / "config.yaml")
        with (run_dir / "history.pickle").open("wb") as stream:
            pickle.dump({
                "train": [0.5] * (len(validation_aucs) - 1),
                "val": [0.5] * len(validation_aucs),
                "auc": validation_aucs,
            }, stream)
        with (run_dir / "trained_model.pickle").open("wb") as stream:
            pickle.dump({
                "weights": CircuitWeights(
                    rot=np.zeros((1, 4, 3)),
                    aux={"scale_factor": 1.0, "bias": 0.1, "hamiltonian_coeffs": [0.1] * 4},
                ),
                "training": {
                    "completed_epochs": len(validation_aucs) - 1,
                    "stop_reason": "early_stopping",
                },
            }, stream)
        (run_dir / "wandb_run_id.txt").write_text(f"run{random_seed}", encoding="utf-8")
        if epoch_times is not None:
            with (run_dir / "epoch_times.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["epoch", "seconds"])
                for epoch, seconds in enumerate(epoch_times, start=1):
                    writer.writerow([epoch, seconds])
        if compile_times is not None:
            with (run_dir / "compile_times.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["step", "seconds"])
                for step, seconds in compile_times.items():
                    writer.writerow([step, seconds])
        with (result_dir / "test_results.pickle").open("wb") as stream:
            pickle.dump({
                "scores": np.asarray(scores),
                "labels": np.asarray(labels),
                "auc": 0.0,
            }, stream)

    def test_publishes_real_shaped_aggregates_and_index_last(self) -> None:
        self._write_run("002", 42, [0.60, 0.70, 0.80], [0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
        self._write_run("002", 43, [0.40, 0.50], [0.1, 0.8, 0.7, 0.9], [0, 0, 1, 1])
        second_config_path = self.models / "002" / "43" / "config.yaml"
        second_config = OmegaConf.load(second_config_path)
        second_config.lr = 0.01
        OmegaConf.save(second_config, second_config_path)

        index = publish_reports.publish_reports(
            self.models, self.results, self.output, self.site,
        )
        detail = json.loads((self.output / "data" / "002.json").read_text())

        self.assertEqual(index["schema_version"], 1)
        self.assertEqual(index["experiments"][0]["id"], "002")
        self.assertEqual(index["experiments"][0]["status"], "complete")
        self.assertEqual(index["experiments"][0]["successful_runs"], 2)
        self.assertAlmostEqual(index["experiments"][0]["mean_auc"], 0.875)
        self.assertAlmostEqual(index["experiments"][0]["std_auc"], 0.125)
        # Both runs evaluate the same underlying test jets (random_seed only
        # reshuffles case_reader's deterministic selection) -- total_jets must
        # report that shared count, not sum duplicate evaluations across runs.
        self.assertEqual(detail["summary"]["total_jets"], 4)
        self.assertEqual(detail["summary"]["jets_per_run"], 4)
        aggregate = detail["validation"]["aggregate"]
        self.assertEqual([point["epoch"] for point in aggregate], [0, 1])
        self.assertEqual([point["mean"] for point in aggregate], [0.5, 0.6])
        for point in aggregate:
            self.assertAlmostEqual(point["std"], 0.1)
        self.assertEqual(len(detail["roc"]["aggregate"]), 1001)
        self.assertEqual(len(detail["roc"]["runs"]), 2)
        self.assertEqual(detail["runs"][0]["completed_epochs"], 2)
        self.assertEqual(detail["runs"][0]["stop_reason"], "early_stopping")
        self.assertEqual(detail["config"]["optimization"]["lr"], "Varies by run")
        self.assertEqual(
            detail["config_variations"]["optimization"]["lr"],
            [
                {"random_seed": 42, "value": 0.05},
                {"random_seed": 43, "value": 0.01},
            ],
        )
        self.assertEqual(
            detail["runs"][0]["wandb_run_id"],
            "run42",
        )
        for relative_path in publish_reports.FRONTEND_FILES:
            self.assertTrue((self.output / relative_path).is_file())
        self.assertTrue((self.output / "data" / "index.json").is_file())
        self.assertEqual(
            stat.S_IMODE((self.output / "data" / "index.json").stat().st_mode),
            0o644,
        )

    def test_epoch_times_aggregate_and_individual_are_published(self) -> None:
        self._write_run(
            "004", 50, [0.5, 0.6, 0.7], [0.1, 0.9], [0, 1],
            epoch_times=[10.0, 12.0, 14.0],
        )
        self._write_run(
            "004", 51, [0.5, 0.6], [0.1, 0.9], [0, 1],
            epoch_times=[8.0, 10.0],
        )
        # No epoch_times.csv for this run -- an old run predating the feature.
        self._write_run("004", 52, [0.5, 0.6], [0.1, 0.9], [0, 1])

        publish_reports.publish_reports(self.models, self.results, self.output, self.site)
        detail = json.loads((self.output / "data" / "004.json").read_text())

        aggregate = detail["epoch_times"]["aggregate"]
        self.assertEqual([point["epoch"] for point in aggregate], [1, 2])
        self.assertEqual([point["mean"] for point in aggregate], [9.0, 11.0])
        for point in aggregate:
            self.assertAlmostEqual(point["std"], 1.0)
            self.assertEqual(point["n"], 2)

        runs = {run["random_seed"]: run for run in detail["epoch_times"]["runs"]}
        self.assertEqual(
            runs[50]["points"],
            [{"epoch": 1, "seconds": 10.0}, {"epoch": 2, "seconds": 12.0}, {"epoch": 3, "seconds": 14.0}],
        )
        self.assertNotIn(52, runs)

        no_csv_run = next(run for run in detail["runs"] if run["random_seed"] == 52)
        self.assertNotIn("epoch_times", no_csv_run)
        messages = [notice["message"] for notice in detail["notices"]]
        self.assertFalse(any("poch tim" in message for message in messages))

    def test_compile_times_are_aggregated_and_excluded_from_public_runs(self) -> None:
        self._write_run(
            "005", 60, [0.5, 0.6], [0.1, 0.9], [0, 1],
            compile_times={"train": 600.0, "val": 120.0},
        )
        self._write_run(
            "005", 61, [0.5, 0.6], [0.1, 0.9], [0, 1],
            compile_times={"train": 620.0, "val": 100.0},
        )
        # No compile_times.csv for this run -- an autograd run or one predating the feature.
        self._write_run("005", 62, [0.5, 0.6], [0.1, 0.9], [0, 1])

        publish_reports.publish_reports(self.models, self.results, self.output, self.site)
        detail = json.loads((self.output / "data" / "005.json").read_text())

        compile_times = detail["compile_times"]
        self.assertAlmostEqual(compile_times["train"]["mean"], 610.0)
        self.assertAlmostEqual(compile_times["train"]["std"], 10.0)
        self.assertEqual(compile_times["train"]["n"], 2)
        self.assertAlmostEqual(compile_times["val"]["mean"], 110.0)
        self.assertAlmostEqual(compile_times["val"]["std"], 10.0)
        self.assertEqual(compile_times["val"]["n"], 2)

        for run in detail["runs"]:
            self.assertNotIn("compile_times", run)
        messages = [notice["message"] for notice in detail["notices"]]
        self.assertFalse(any("compile" in message.lower() for message in messages))

    def test_missing_compile_times_produces_empty_aggregate(self) -> None:
        self._write_run("006", 70, [0.5, 0.6], [0.1, 0.9], [0, 1])

        publish_reports.publish_reports(self.models, self.results, self.output, self.site)
        detail = json.loads((self.output / "data" / "006.json").read_text())

        self.assertEqual(detail["compile_times"], {})

    def test_jax_backend_experiment_publishes_without_error(self) -> None:
        """publish_reports only reads config.yaml/results, never a checkpoint, so it
        should be backend-agnostic already -- confirmed here rather than assumed."""
        self._write_run("003", 44, [0.5, 0.6], [0.1, 0.9], [0, 1])
        config_path = self.models / "003" / "44" / "config.yaml"
        config = OmegaConf.load(config_path)
        config.backend = "jax"
        OmegaConf.save(config, config_path)

        publish_reports.publish_reports(self.models, self.results, self.output, self.site)
        detail = json.loads((self.output / "data" / "003.json").read_text())
        self.assertEqual(detail["config"]["execution"]["backend"], "jax")

    def test_redacts_paths_and_preserves_missing_artifact_notices(self) -> None:
        self._write_run("001", 40, [0.5, 0.6], [0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
        incomplete = self.models / "001" / "41"
        incomplete.mkdir(parents=True)
        (incomplete / "config.yaml").write_text("not: [valid", encoding="utf-8")
        (self.models / "001" / "notes").mkdir()
        orphan = self.results / "001" / "99"
        orphan.mkdir(parents=True)

        publish_reports.publish_reports(
            self.models, self.results, self.output, self.site,
        )
        detail_text = (self.output / "data" / "001.json").read_text()
        detail = json.loads(detail_text)
        index = json.loads((self.output / "data" / "index.json").read_text())

        self.assertNotIn(str(self.root), detail_text)
        self.assertNotIn("data_dir", detail_text)
        self.assertNotIn("save_dir", detail_text)
        self.assertNotIn("dump", detail_text)
        self.assertEqual(detail["experiment"]["status"], "partial")
        self.assertEqual(detail["summary"]["run_count"], 2)
        self.assertEqual(detail["summary"]["successful_runs"], 1)
        missing_run = next(run for run in detail["runs"] if run["random_seed"] == 41)
        self.assertEqual(missing_run["status"], "unavailable")
        messages = [notice["message"] for notice in detail["notices"]]
        self.assertTrue(any("Configuration is malformed" in message for message in messages))
        self.assertTrue(any("Missing evaluation result" in message for message in messages))
        self.assertTrue(any("unknown random seed 99" in message for message in messages))
        self.assertTrue(any("nonnumeric saved-run" in message for message in messages))
        self.assertEqual(index["experiments"][0]["status"], "partial")
        self.assertTrue(index["experiments"][0]["notices"])

    def test_circuit_diagram_is_generated_and_aux_weights_are_published(self) -> None:
        self._write_run("007", 80, [0.5, 0.6], [0.1, 0.9], [0, 1])
        self._write_run("007", 81, [0.5, 0.6], [0.1, 0.9], [0, 1])

        publish_reports.publish_reports(self.models, self.results, self.output, self.site)
        detail = json.loads((self.output / "data" / "007.json").read_text())

        self.assertEqual(detail["circuit_diagram"], "data/circuits/007.png")
        diagram_path = self.output / "data" / "circuits" / "007.png"
        self.assertTrue(diagram_path.is_file())
        self.assertGreater(diagram_path.stat().st_size, 0)

        run = detail["runs"][0]
        self.assertEqual(run["aux_weights"]["scale_factor"], 1.0)
        self.assertEqual(run["aux_weights"]["bias"], 0.1)
        self.assertEqual(run["aux_weights"]["hamiltonian_coeffs"], [0.1, 0.1, 0.1, 0.1])
        messages = [notice["message"] for notice in detail["notices"]]
        self.assertFalse(any("circuit diagram" in message.lower() for message in messages))

    def test_missing_circuit_snapshot_reports_a_notice(self) -> None:
        self._write_run("008", 90, [0.5, 0.6], [0.1, 0.9], [0, 1])
        shutil.rmtree(self.models / "008" / "90" / "circuits")

        publish_reports.publish_reports(self.models, self.results, self.output, self.site)
        detail = json.loads((self.output / "data" / "008.json").read_text())

        self.assertIsNone(detail["circuit_diagram"])
        messages = [notice["message"] for notice in detail["notices"]]
        self.assertTrue(any("circuit diagram" in message.lower() for message in messages))


if __name__ == "__main__":
    unittest.main()
