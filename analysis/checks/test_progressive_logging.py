"""No-network logger checks, including one real W&B offline run."""
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from ga_search.progressive_logging import ExperimentLogger


class LoggerTests(unittest.TestCase):
    def test_disabled_sanitization_resume_and_threads_without_sdk(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("ga_search.progressive_logging.importlib.import_module", side_effect=AssertionError("SDK imported")):
                logger = ExperimentLogger({"api_key": "test-credential"}, tmp)
                logger.emit("epoch_end", loss=torch.tensor(0.5), invalid=np.float64(np.nan), vector=np.array([1, np.inf]))
                threads = [threading.Thread(target=logger.emit, args=("epoch_end",), kwargs={"epoch": i}) for i in range(5)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                logger.finish()
                resumed = ExperimentLogger({}, tmp)
                self.assertEqual(logger.run_id, resumed.run_id)
                resumed.emit("search_resume")
                resumed.finish()
            rows = [json.loads(line) for line in (Path(tmp) / "events.jsonl").read_text().splitlines()]
            self.assertEqual([row["sequence"] for row in rows], list(range(len(rows))))
            self.assertIsNone(rows[0]["invalid"])
            self.assertEqual(rows[0]["vector"], [1, None])
            self.assertEqual(rows[0]["loss"], 0.5)
            self.assertFalse(logger.enabled)
            self.assertIsNone(logger.url)

    def test_valid_only_generation_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = ExperimentLogger({}, tmp)
            row = logger.emit("generation_end", generation=0, population=[
                {"result": {"status": "ok", "J": 0.4}},
                {"result": {"status": "ok", "J": 0.6}},
                {"result": {"status": "invalid", "J": 10}},
            ])
            summary = row["generation_summary"]
            self.assertEqual(summary["ga/J_mean"], 0.5)
            self.assertEqual(summary["ga/J_min"], 0.4)
            self.assertEqual(summary["ga/invalid_count"], 1)
            invalid = logger.emit("generation_end", generation=1, population=[{"result": {"status": "invalid", "J": 10}}])
            self.assertIsNone(invalid["generation_summary"]["ga/J_min"])
            self.assertEqual(invalid["generation_summary"]["ga/best_valid_J"], 0.4)
            logger.finish()

    def test_real_offline_sdk_artifact_redaction_and_identity_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.enterContext(patch.dict(os.environ, {
                "WANDB_CACHE_DIR": str(root / "cache"),
                "WANDB_CONFIG_DIR": str(root / "settings"),
                "WANDB_DATA_DIR": str(root / "sdk-data"),
            }))
            config = {"seed": 42, "api_key": "do-not-upload", "wandb": {
                "enabled": True, "mode": "offline", "project": "progressive_nas_ga", "entity": "ophira-ai"}}
            (root / "config.json").write_text(json.dumps(config))
            (root / "data_identity.json").write_text('{"sha256":"fixture"}')
            (root / "checkpoint.pt").write_bytes(b"not-an-artifact")
            logger = ExperimentLogger(config, root)
            logger.emit("individual_result", generation=0, individual_index=0,
                        requested_genome={"blocks": [{"channels": 4}]},
                        effective_genome={"blocks": [{"channels": 4}]},
                        result={"status": "ok", "J": 0.3, "dbi": 0.4, "cost": {"parameters": 20}})
            logger.emit("generation_end", generation=0)
            run_id = logger.run_id
            logger.finish()
            self.assertTrue(logger.enabled)
            self.assertTrue(list((root / "wandb").glob("offline-run-*")))
            snapshots = list((root / "artifact_snapshots").glob("*/config.json"))
            self.assertTrue(snapshots)
            self.assertTrue(all(json.loads(path.read_text())["api_key"] == "[REDACTED]" for path in snapshots))
            self.assertFalse(list((root / "artifact_snapshots").rglob("checkpoint.pt")))
            resumed = ExperimentLogger(config, root)
            self.assertEqual(resumed.run_id, run_id)
            self.assertEqual(resumed.emit("search_resume")["sequence"], 3)
            resumed.finish()

    def test_online_init_error_is_not_silently_downgraded(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("ga_search.progressive_logging.importlib.import_module", side_effect=RuntimeError("init unavailable")):
                with self.assertRaisesRegex(RuntimeError, "init unavailable"):
                    ExperimentLogger({"wandb": {"enabled": True, "mode": "online"}}, tmp)


if __name__ == "__main__":
    unittest.main()
