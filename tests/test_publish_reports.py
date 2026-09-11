"""
Verify static report publication, aggregation, and artifact failure handling.

Author: Aritra Bal (ETP)
Date: 2026-09-10
"""

import json
import stat
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
from omegaconf import OmegaConf

import publish_reports


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
    ) -> None:
        run_dir = self.models / experiment / str(random_seed)
        result_dir = self.results / experiment / str(random_seed)
        run_dir.mkdir(parents=True)
        result_dir.mkdir(parents=True)
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
                "training": {
                    "completed_epochs": len(validation_aucs) - 1,
                    "stop_reason": "early_stopping",
                },
            }, stream)
        (run_dir / "wandb_run_id.txt").write_text(f"run{random_seed}", encoding="utf-8")
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
        self.assertEqual(detail["summary"]["total_jets"], 8)
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


if __name__ == "__main__":
    unittest.main()
