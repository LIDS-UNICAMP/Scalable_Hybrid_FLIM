"""Monitor fixtures and injected API; no network or W&B SDK initialization."""
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from ga_search.progressive_monitor import Monitor


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nas-monitor-tests-")
        self.root = Path(self.temp.name) / "run"
        self.root.mkdir()
        self.now = 2000.0

    def tearDown(self):
        self.temp.cleanup()

    def write(self, path, value):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value))

    def monitor(self, **kwargs):
        return Monitor([self.root], report=Path(self.temp.name) / "report.md",
                       status=Path(self.temp.name) / "status.json", now=lambda: self.now, **kwargs)

    def events(self, rows):
        for i, row in enumerate(rows):
            row.setdefault("sequence", i)
            row.setdefault("time", datetime.fromtimestamp(1000 + i, timezone.utc).isoformat())
        (self.root / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_unique_architectures_penalty_not_best_and_l1_not_fitness(self):
        good = {"status": "ok", "J": .3, "dbi": .4, "hoyer_mean": .2, "coverage": .9, "depth": 2}
        genome = {"blocks": [{"kernel": 3}]}
        rows = [{"event": "individual_result", "requested_genome": genome, "result": good},
                {"event": "individual_result", "requested_genome": genome, "cache": True, "result": good},
                {"event": "individual_result", "requested_genome": {"blocks": []},
                 "result": {"status": "infeasible", "J": 1e12, "reason": "MAC limit"}},
                {"event": "epoch_end", "individual_id": "one", "depth": 2,
                 "result": {"reconstruction_loss": .4, "J": .3}},
                {"event": "epoch_end", "individual_id": "one", "depth": 2,
                 "result": {"reconstruction_loss": .2, "J": .35}}]
        self.events(rows)
        self.write("logger_state.json", {"sequence": 4, "enabled": False})
        result = self.monitor(remote=False).poll()["roots"][0]
        self.assertEqual(result["unique_architectures"], 2)
        self.assertEqual(result["cached_or_retained_events"], 1)
        self.assertEqual(result["invalid_rate"], .5)
        self.assertEqual(result["best_valid"]["J"], .3)
        self.assertTrue(any("L1 caiu" in alert for alert in result["alerts"]))
        self.assertTrue(any("Sem evento" in alert for alert in result["alerts"]))
        self.assertTrue((Path(self.temp.name) / "report.md").exists())

    def test_remote_verified_sequence_lag_artifacts_and_safe_error(self):
        self.events([{"event": "generation_end"}])
        self.write("logger_state.json", {"sequence": 10, "enabled": True,
                                         "url": "https://wandb.ai/entity/project/runs/abc?token=do-not-show"})
        artifact = SimpleNamespace(type="nas-logs")
        run = SimpleNamespace(state="running", summary={"_step": 5, "_timestamp": 1000},
                              logged_artifacts=lambda: [artifact])
        api = SimpleNamespace(run=lambda path: run)
        monitor = self.monitor(api=api)
        result = monitor.poll()["roots"][0]
        self.assertTrue(result["remote"]["verified"])
        self.assertEqual(result["remote"]["sequence_lag"], 5)
        self.assertEqual(result["remote"]["nas_log_artifacts"], 1)
        self.assertNotIn("token", result["remote"]["url"])
        self.assertTrue(any("tolerância" in alert for alert in result["alerts"]))
        def error(path):
            raise RuntimeError("credential=never-print-this")
        api.run = error
        for _ in range(3):
            result = monitor.poll()["roots"][0]
        self.assertEqual(result["remote"]["error_polls"], 3)
        self.assertFalse(result["remote"]["verified"])
        self.assertNotIn("never-print", json.dumps(result))
        self.assertTrue(any("três consultas" in alert for alert in result["alerts"]))

    def test_missing_roots_partial_json_and_finished_job(self):
        monitor = self.monitor(remote=False)
        row = monitor.poll()["roots"][0]
        self.assertFalse(row["terminal"])
        (self.root / "events.jsonl").write_text('{"event":')
        (self.root / "exit_code.txt").write_text("0")
        self.write("search/search_state.json", {"generation": 5, "finished": True})
        row = monitor.poll()["roots"][0]
        self.assertEqual(row["malformed_event_lines"], 1)
        self.assertTrue(row["terminal"])
        self.assertEqual(row["generation_completed"], 5)


if __name__ == "__main__":
    unittest.main()
