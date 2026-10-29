"""Genetic operators and restart/cache contracts, without training surrogates."""
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path

from ga_search.progressive_genetics import (
    ArchitectureOperators, EvaluationCache, GeneticSearch, first_changed_prefix,
    validate_genome,
)


def score(genome):
    # Deterministic injected objective tests search mechanics, not model quality.
    return {"J": sum(b["channels"] + b["kernel"] / 10 for b in genome["blocks"]),
            "status": "valid"}


class GeneticsTests(unittest.TestCase):
    def config(self):
        return {"seed": 17, "min_depth": 1, "max_depth": 4,
                "population_size": 4, "elitism": 1, "generations": 3,
                "mutation_rate": 1.0, "patience": 4}

    def test_operators_valid_and_independent(self):
        ops = ArchitectureOperators(self.config(), random.Random(17))
        parent = ops.sample()
        original = copy.deepcopy(parent)
        observed_insert = observed_remove = observed_change = False
        for _ in range(200):
            a, b = ops.sample(), ops.sample()
            snapshot = copy.deepcopy(a)
            child = ops.mutate(a)
            validate_genome(child, 4)
            self.assertEqual(a, snapshot)
            observed_insert |= len(child["blocks"]) > len(a["blocks"])
            observed_remove |= len(child["blocks"]) < len(a["blocks"])
            observed_change |= len(child["blocks"]) == len(a["blocks"]) and child != a
            crossed = ops.crossover(a, b)
            validate_genome(crossed, 4)
            crossed["blocks"][0]["channels"] = 999
            self.assertEqual(a, snapshot)
        self.assertTrue(observed_insert and observed_remove and observed_change)
        self.assertEqual(parent, original)
        for config in ({"max_depth": 11}, {"min_depth": 3, "max_depth": 2},
                       {"space": {"kernel": [1]}}, {"space": {"groups": [4]}}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                ArchitectureOperators(config, random.Random(1))

    def test_genome_rejection_and_prefix(self):
        ops = ArchitectureOperators(self.config(), random.Random(2))
        parent = {"blocks": [ops.block(), ops.block(), ops.block()]}
        for key, value in (("op", "depthwise"), ("connection", "concat"),
                           ("connection", "residual"), ("kernel", 1),
                           ("channels", 0), ("channels", 2.5), ("pool_stride", 3),
                           ("bn", 1), ("dropout", 1), ("dropout", -0.1),
                           ("activation", "gelu")):
            candidate = copy.deepcopy(parent)
            candidate["blocks"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_genome(candidate)
        for blocks in ([], parent["blocks"] * 4):
            with self.assertRaises(ValueError):
                validate_genome({"blocks": blocks})
        child = copy.deepcopy(parent)
        child["blocks"][1]["kernel"] = 3 if parent["blocks"][1]["kernel"] == 5 else 5
        self.assertEqual(first_changed_prefix(parent, child), 1)
        self.assertEqual(first_changed_prefix(parent, parent), 3)
        self.assertEqual(first_changed_prefix(parent, {"blocks": parent["blocks"][:2]}), 2)
        child["blocks"][0]["channels"] = 128
        self.assertEqual(first_changed_prefix(parent, child), 0)

    def test_cache_all_identity_fields(self):
        context = {"effective_genome": {"blocks": [{"channels": 4}]},
                   "data": {"dataset": "synthetic", "split": "train1/val1", "content_sha256": "abc"},
                   "config": self.config(), "seeds": [17], "sample_ids": ["train/0"],
                   "versions": {"torch": "test-version"}, "prefix_id": "checkpoint-a"}
        with tempfile.TemporaryDirectory() as directory:
            cache = EvaluationCache(directory)
            cache.put(context, {"J": .5})
            self.assertEqual(cache.get(context), {"J": .5})
            for field in context:
                changed = copy.deepcopy(context)
                changed[field] = {"changed": True}
                with self.subTest(field=field):
                    self.assertNotEqual(cache.key(context), cache.key(changed))
                    self.assertIsNone(cache.get(changed))
                missing = copy.deepcopy(context)
                del missing[field]
                with self.assertRaises(ValueError):
                    cache.key(missing)
            prefix_b = {**context, "prefix_id": "checkpoint-b"}
            cache.put(prefix_b, {"J": .75})
            self.assertEqual(cache.get(context)["J"], .5)
            self.assertEqual(cache.get(prefix_b)["J"], .75)
            changed_split = copy.deepcopy(context)
            changed_split["data"]["split"] = "train2/val2"
            self.assertIsNone(cache.get(changed_split))

    def test_generations_elitism_determinism_and_resume(self):
        config = self.config()
        context = {"dataset": "synthetic", "split": "train/validation"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uninterrupted = GeneticSearch(config, score, root / "full", context).run()
            repeat = GeneticSearch(config, score, root / "repeat", context).run()
            self.assertEqual(uninterrupted, repeat)
            self.assertTrue(uninterrupted["finished"])
            self.assertEqual(len(uninterrupted["history"]), 3)
            history = uninterrupted["history"]
            for previous, current in zip(history, history[1:]):
                self.assertLessEqual(current["population"][0]["result"]["J"],
                                     previous["population"][0]["result"]["J"])
                self.assertIn(previous["population"][0], current["population"])
            evaluations = []

            def interrupted(genome):
                if len(evaluations) == 2:
                    raise RuntimeError("simulated interruption at evaluation boundary")
                evaluations.append(copy.deepcopy(genome))
                return score(genome)

            search = GeneticSearch(config, interrupted, root / "resume", context)
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                search.run()
            disk = json.loads(search.path.read_text())
            self.assertEqual(sum(i["result"] is not None for i in disk["population"]), 2)
            resumed_calls = []

            def resumed_score(genome):
                resumed_calls.append(copy.deepcopy(genome))
                return score(genome)

            resumed = GeneticSearch(config, resumed_score, root / "resume", context).run(resume=True)
            self.assertEqual(resumed, uninterrupted)
            # 4 initial + 3 children in each of two generations, minus two saved evaluations.
            self.assertEqual(len(resumed_calls), 8)
            self.assertEqual(GeneticSearch(config, resumed_score, root / "resume", context).run(resume=True),
                             json.loads(json.dumps(resumed)))
            with self.assertRaises(FileExistsError):
                GeneticSearch(config, score, root / "full", context).run()
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                GeneticSearch(config, score, root / "resume", {"dataset": "different"}).run(resume=True)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                GeneticSearch({**config, "seed": 99}, score, root / "resume", context).run(resume=True)

    def test_logger_events_do_not_change_search_or_rng(self):
        class RecordingLogger:
            def __init__(self):
                self.events = []

            def emit(self, event, **payload):
                self.events.append((event, payload))

        config = {**self.config(), "generations": 2}
        def objective():
            calls = 0
            def evaluate(genome):
                nonlocal calls
                calls += 1
                if calls % 3 == 0:
                    return {"J": 1e12, "status": "invalid", "reason": "fixture: low coverage"}
                return {**score(genome), "cache_hit": calls % 2 == 0}
            return evaluate

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = GeneticSearch(config, objective(), root / "plain").run()
            logger = RecordingLogger()
            logged = GeneticSearch(config, objective(), root / "logged", logger=logger).run()
            self.assertEqual(logged, baseline)
            self.assertEqual((root / "plain/search_state.json").read_bytes(),
                             (root / "logged/search_state.json").read_bytes())
            names = [event for event, _ in logger.events]
            self.assertEqual(names[0], "search_start")
            self.assertEqual(names[-1], "search_end")
            self.assertEqual(names.count("generation_start"), 2)
            self.assertEqual(names.count("generation_end"), 2)
            self.assertEqual(names.count("individual_start"), 8)
            self.assertEqual(names.count("individual_result"), 8)
            self.assertEqual(names.count("offspring"), 3)
            self.assertIn("best_update", names)
            results = [payload for event, payload in logger.events if event == "individual_result"]
            self.assertTrue(any(payload["cache"] for payload in results))
            self.assertTrue(any(payload["status"] == "invalid" and payload["reason"] for payload in results))
            self.assertTrue(all(payload["elapsed_seconds"] >= 0 for payload in results))
            self.assertEqual([payload["individual_index"] for payload in results[:4]], [0, 1, 2, 3])
            for event, payload in logger.events:
                if event == "generation_end":
                    valid = [item["result"]["J"] for item in payload["population"]
                             if item["result"]["status"] == "valid"]
                    summary = payload["summary"]
                    self.assertEqual(summary["feasible_count"], len(valid))
                    self.assertEqual(summary["invalid_count"], 4 - len(valid))
                    self.assertAlmostEqual(summary["mean_J"], sum(valid) / len(valid))
                    self.assertLess(summary["mean_J"], 1e12)
                elif event == "offspring":
                    self.assertEqual(payload["generation"], 1)
                    self.assertEqual(len(payload["parents"]), 2)
                    self.assertTrue(payload["filters_invalidated"])
                    self.assertFalse(payload["weight_inheritance"])
                    self.assertTrue(payload["mutation"]["applied"])
                    for index, parent in enumerate(payload["parents"]):
                        self.assertEqual(payload["first_changed_prefix"][index],
                                         first_changed_prefix(parent["requested_genome"], payload["requested_genome"]))
            resumed_logger = RecordingLogger()
            resumed = GeneticSearch(config, score, root / "logged", logger=resumed_logger).run(resume=True)
            self.assertEqual(resumed, json.loads(json.dumps(logged)))
            self.assertEqual([event for event, _ in resumed_logger.events], ["search_resume", "search_end"])

    def test_logger_snapshot_isolation_and_all_invalid_summary(self):
        class MutatingLogger:
            def __init__(self):
                self.summaries = []

            def emit(self, event, **payload):
                if event == "generation_end":
                    self.summaries.append(copy.deepcopy(payload["summary"]))
                # Even a callback accidentally editing its arguments cannot corrupt search.
                if "population" in payload:
                    payload["population"].clear()
                if "requested_genome" in payload:
                    payload["requested_genome"]["blocks"].clear()

        config = {**self.config(), "generations": 2}
        logger = MutatingLogger()
        with tempfile.TemporaryDirectory() as directory:
            state = GeneticSearch(config, lambda genome: {"J": 1e12, "status": "infeasible", "reason": "fixture"},
                                  directory, logger=logger).run()
        self.assertTrue(state["finished"])
        self.assertEqual(len(state["history"]), 2)
        for summary in logger.summaries:
            self.assertEqual(summary["invalid_count"], 4)
            self.assertEqual(summary["feasible_count"], 0)
            self.assertIsNone(summary["mean_J"])
            self.assertIsNone(summary["std_J"])
            self.assertIsNone(summary["best_valid_J"])

    def test_invalid_result_is_not_silent(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "finite J"):
                GeneticSearch(self.config(), lambda genome: {"J": float("nan")}, directory).run()


if __name__ == "__main__":
    unittest.main()
