"""Scalar genetic search for the opt-in progressive SPiFiL experiment.

Individuals never inherit weights. Every changed prefix is fitted again. This
module owns architecture operators only; the growth runner supplies evaluation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validate_genome(genome, max_depth=10):
    blocks = genome.get("blocks", [])
    if not 1 <= len(blocks) <= min(10, max_depth):
        raise ValueError("evolutionary depth must be between 1 and max_depth <= 10")
    for b in blocks:
        if b.get("op", "conv") != "conv" or b.get("connection", "sequential") != "sequential":
            raise ValueError("initial SPiFiL space supports sequential conventional convolutions only")
        if b.get("kernel") not in (3, 5) or b.get("pool_stride") not in (1, 2):
            raise ValueError("supported kernel=3|5 and pool_stride=1|2")
        if type(b.get("channels")) is not int or b["channels"] <= 0:
            raise ValueError("channels must be a positive integer")
        if not isinstance(b.get("bn", False), bool) or not 0 <= b.get("dropout", 0) < 1:
            raise ValueError("invalid normalization or dropout")
        if b.get("activation", "relu") not in ("relu", "relu6", "silu"):
            raise ValueError("unsupported activation")
    return genome


def first_changed_prefix(parent, child):
    for i, (a, b) in enumerate(zip(parent["blocks"], child["blocks"])):
        if a != b:
            return i
    return min(len(parent["blocks"]), len(child["blocks"]))


DEFAULT_SPACE = dict(kernel=[3, 5], channels=[4, 8, 16], pool_stride=[2],
                     bn=[False, True], dropout=[0.0, 0.1], activation=["relu", "relu6", "silu"])


class ArchitectureOperators:
    def __init__(self, config, rng):
        self.config, self.rng = config, rng
        self.max_depth = int(config.get("max_depth", 10))
        self.min_depth = int(config.get("min_depth", 1))
        if not 1 <= self.min_depth <= self.max_depth <= 10:
            raise ValueError("require 1 <= min_depth <= max_depth <= 10")
        self.space = {**DEFAULT_SPACE, **config.get("space", {})}
        unknown = set(self.space) - set(DEFAULT_SPACE)
        if unknown or any(not isinstance(v, list) or not v for v in self.space.values()):
            raise ValueError("invalid search space axes")
        # Validate every axis choice before evaluation starts.
        base = {k: v[0] for k, v in self.space.items()}
        for k, values in self.space.items():
            for value in values:
                validate_genome({"blocks": [{**base, k: value}]})

    def block(self):
        return dict(op="conv", connection="sequential",
                    **{k: self.rng.choice(v) for k, v in self.space.items()})

    def sample(self):
        return {"blocks": [self.block() for _ in range(self.rng.randint(self.min_depth, self.max_depth))]}

    def mutate(self, genome):
        child = copy.deepcopy(genome)
        blocks = child["blocks"]
        actions = ["change"]
        if len(blocks) < self.max_depth:
            actions.append("insert")
        if len(blocks) > self.min_depth:
            actions.append("remove")
        action = self.rng.choice(actions)
        i = self.rng.randrange(len(blocks))
        if action == "insert":
            blocks.insert(i, self.block())
        elif action == "remove":
            blocks.pop(i)
        else:
            k = self.rng.choice(list(self.space))
            alternatives = [v for v in self.space[k] if v != blocks[i][k]]
            if alternatives:
                blocks[i][k] = self.rng.choice(alternatives)
        return validate_genome(child, self.max_depth)

    def crossover(self, a, b):
        i, j = self.rng.randint(0, len(a["blocks"])), self.rng.randint(0, len(b["blocks"]))
        blocks = copy.deepcopy(a["blocks"][:i] + b["blocks"][j:])[:self.max_depth]
        while len(blocks) < self.min_depth:
            blocks.append(self.block())
        return validate_genome({"blocks": blocks}, self.max_depth)


CACHE_FIELDS = ("effective_genome", "data", "config", "seeds", "sample_ids", "versions", "prefix_id")


class EvaluationCache:
    """Exact context cache; architecture alone can never identify an evaluation."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(context):
        missing = set(CACHE_FIELDS) - set(context)
        if missing:
            raise ValueError(f"cache context missing {sorted(missing)}")
        return digest(context)

    def get(self, context):
        path = self.directory / (self.key(context) + ".json")
        return json.loads(path.read_text()) if path.exists() else None

    def put(self, context, result):
        path = self.directory / (self.key(context) + ".json")
        atomic_json(path, result)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    tmp.replace(path)


class GeneticSearch:
    """Tournament/crossover/mutation/elitism with restart at evaluation boundaries."""
    def __init__(self, config, evaluate, directory, context=None, logger=None):
        self.config, self.evaluate = config, evaluate
        self.logger = logger
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "search_state.json"
        self.context = digest(context or {})
        self.rng = random.Random(config.get("seed", 42))
        self.ops = ArchitectureOperators(config, self.rng)
        self.population_size = int(config.get("population_size", 4))
        self.elites = int(config.get("elitism", 1))
        if self.population_size < 2 or not 1 <= self.elites < self.population_size:
            raise ValueError("population_size >= 2 and 1 <= elitism < population_size required")

    def _emit(self, event, **payload):
        # Loggers receive snapshots, so callbacks cannot mutate search state.
        # Logging does not draw from the search RNG or add checkpoint fields.
        if self.logger is not None:
            self.logger.emit(event, **copy.deepcopy(payload))

    @staticmethod
    def _summary(ranked):
        valid = [float(item["result"]["J"]) for item in ranked
                 if item["result"].get("status", "ok") not in
                 {"invalid", "infeasible", "failed", "error"}]
        mean = statistics.mean(valid) if valid else None
        std = statistics.pstdev(valid) if valid else None
        return dict(best_J=float(ranked[0]["result"]["J"]),
                    best_valid_J=min(valid) if valid else None,
                    mean_J=mean, std_J=std,
                    feasible_count=len(valid), invalid_count=len(ranked) - len(valid),
                    population_size=len(ranked))

    def _save(self, state):
        state["rng"] = self.rng.getstate()
        atomic_json(self.path, state)

    def run(self, resume=False):
        def tuples(value):
            return tuple(tuples(x) for x in value) if isinstance(value, list) else value
        resuming = resume and self.path.exists()
        if resuming:
            state = json.loads(self.path.read_text())
            if state["context"] != self.context or state["config"] != self.config:
                raise ValueError("resume configuration/data identity mismatch")
            self.rng.setstate(tuples(state["rng"]))
        else:
            if self.path.exists():
                raise FileExistsError("search exists; use resume")
            state = dict(context=self.context, config=self.config, generation=0, history=[],
                         best_J=None, stale=0, population=[dict(genome=self.ops.sample(), result=None)
                                                        for _ in range(self.population_size)], finished=False)
            self._save(state)
        self._emit("search_resume" if resuming else "search_start",
                   config=self.config, generation=state["generation"],
                   population=state["population"], finished=state["finished"])
        if state["finished"]:
            self._emit("search_end", generation=max(0, state["generation"] - 1),
                       generations=state["generation"], best=state.get("best"),
                       best_J=state["best_J"], status="finished", resumed=True)
            return state
        generations = int(self.config.get("generations", 2))
        if generations < 1:
            raise ValueError("generations must be positive")
        while state["generation"] < generations:
            generation = state["generation"]
            generation_started = time.perf_counter()
            self._emit("generation_start", generation=generation,
                       population=state["population"], population_size=self.population_size)
            for index, individual in enumerate(state["population"]):
                identity = dict(generation=generation, individual_index=index,
                                genome_id=digest(individual["genome"]),
                                requested_genome=individual["genome"])
                reused = individual["result"] is not None
                self._emit("individual_start", **identity, cache=reused,
                           evaluation_required=not reused)
                started = time.perf_counter()
                if not reused:
                    result = self.evaluate(copy.deepcopy(individual["genome"]))
                    if not math.isfinite(float(result["J"])):
                        raise ValueError("evaluator must return finite J, including infeasible candidates")
                    individual["result"] = result
                    self._save(state)
                result = individual["result"]
                event_result = {"status": "ok", "reason": "", **result}
                self._emit("individual_result", **identity, result=event_result,
                           J=result["J"], status=result.get("status", "ok"),
                           reason=result.get("reason", ""),
                           cache=reused or bool(result.get("cache_hit", False)),
                           cache_source="retained_result" if reused else result.get("cache_source"),
                           elapsed_seconds=0.0 if reused else time.perf_counter() - started)
            ranked = sorted(state["population"], key=lambda x: x["result"]["J"])
            score = float(ranked[0]["result"]["J"])
            improved = state["best_J"] is None or score < state["best_J"] - self.config.get("tolerance", 0.0)
            state["stale"] = 0 if improved else state["stale"] + 1
            if improved:
                previous_best = state["best_J"]
                state["best_J"], state["best"] = score, copy.deepcopy(ranked[0])
                self._emit("best_update", generation=generation,
                           genome_id=digest(ranked[0]["genome"]),
                           requested_genome=ranked[0]["genome"], result=ranked[0]["result"],
                           previous_best_J=previous_best, best_J=score)
            self._emit("generation_end", generation=generation, population=ranked,
                       summary=self._summary(ranked), best_J=state["best_J"],
                       elapsed_seconds=time.perf_counter() - generation_started)
            state["history"].append(dict(generation=state["generation"], population=copy.deepcopy(ranked)))
            state["generation"] += 1
            if state["generation"] >= generations or state["stale"] >= self.config.get("patience", generations):
                state["finished"] = True
                self._save(state)
                break
            children = copy.deepcopy(ranked[:self.elites])
            def tournament():
                return min(self.rng.sample(ranked, min(3, len(ranked))), key=lambda x: x["result"]["J"])["genome"]
            while len(children) < self.population_size:
                parent_a, parent_b = tournament(), tournament()
                child = self.ops.crossover(parent_a, parent_b)
                crossed = copy.deepcopy(child)
                mutation_applied = self.rng.random() < self.config.get("mutation_rate", 0.8)
                if mutation_applied:
                    child = self.ops.mutate(child)
                # Structural changes always start at an empty prefix, with no inherited filters.
                self._emit("offspring", generation=generation + 1,
                           individual_index=len(children), genome_id=digest(child),
                           requested_genome=child,
                           parents=[dict(genome_id=digest(parent), requested_genome=parent)
                                    for parent in (parent_a, parent_b)],
                           crossover=dict(applied=True, changed=crossed != parent_a or crossed != parent_b,
                                          first_changed_prefix=[first_changed_prefix(parent, crossed)
                                                                for parent in (parent_a, parent_b)]),
                           mutation=dict(applied=mutation_applied, changed=child != crossed,
                                         operation=("insert" if len(child["blocks"]) > len(crossed["blocks"])
                                                    else "remove" if len(child["blocks"]) < len(crossed["blocks"])
                                                    else "change" if child != crossed else "none"),
                                         blocks_before=len(crossed["blocks"]), blocks_after=len(child["blocks"]),
                                         first_changed_prefix=first_changed_prefix(crossed, child)),
                           first_changed_prefix=[first_changed_prefix(parent, child)
                                                 for parent in (parent_a, parent_b)],
                           filters_invalidated=True, weight_inheritance=False)
                children.append(dict(genome=child, result=None))
            state["population"] = children
            self._save(state)
        self._emit("search_end", generation=max(0, state["generation"] - 1),
                   generations=state["generation"], best=state.get("best"),
                   best_J=state["best_J"], status="finished", resumed=resuming)
        return state
