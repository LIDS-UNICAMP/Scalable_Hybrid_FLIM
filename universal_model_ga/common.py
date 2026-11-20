"""common.py -- utilitários do universal_model_ga: config, caminhos, jsonl incremental, cache por chave e monitor de recursos."""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

PKG = Path(__file__).resolve().parent
NATURAL = ["flowers", "remote_sensing", "ham10000", "aid", "coconut_trees"]
MISSING = None  # campo ausente fica None no json e "AUSENTE" nas tabelas -- nunca inventado


def load_config(path: str | os.PathLike | None = None, **over) -> dict:
    cfg = yaml.safe_load(Path(path or PKG / "config" / "default.yaml").read_text())
    for k, v in over.items():
        if v is not None:
            cfg[k] = v
    for k in ("out_dir", "report_dir"):
        p = Path(cfg[k].format(run_name=cfg["run_name"]))
        cfg[k] = str(p if p.is_absolute() else Path(cfg["repo_root"]) / p)
        Path(cfg[k]).mkdir(parents=True, exist_ok=True)
    q = cfg["cross_eval"]["queue_file"].format(run_name=cfg["run_name"])
    cfg["cross_eval"]["queue_file"] = str(Path(cfg["repo_root"]) / q)
    return cfg


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def key(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def read_jsonl(path) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # linha truncada por interrupção: ignorada (retomada refaz)
    return out


def append_jsonl(path, row: dict):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(row, default=_js) + "\n"); f.flush(); os.fsync(f.fileno())


def write_json(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp"); tmp.write_text(json.dumps(obj, indent=1, default=_js)); tmp.replace(p)


def _js(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, Path): return str(o)
    return str(o)


class Cache:
    """Cache jsonl por chave (arquitetura, base, configuração). Retomada: o que já tem chave não é refeito."""

    def __init__(self, path):
        self.path = Path(path)
        self.rows = {r["_key"]: r for r in read_jsonl(self.path) if "_key" in r}

    def get(self, k):
        return self.rows.get(k)

    def put(self, k, row: dict):
        row = dict(row, _key=k); self.rows[k] = row; append_jsonl(self.path, row); return row


class ResourceMonitor:
    """Amostra CPU %, RSS e (se houver) memória de GPU a cada `every` s; grava resumo por etapa em resources.jsonl."""

    def __init__(self, out_jsonl, stage: str, every: float = 2.0):
        self.out, self.stage, self.every = out_jsonl, stage, every
        self.samples, self._stop = [], threading.Event()

    def _loop(self):
        import psutil
        proc = psutil.Process()
        while not self._stop.is_set():
            gpu = None
            try:
                import torch
                if torch.cuda.is_available():
                    gpu = torch.cuda.max_memory_allocated() / 2**30
            except Exception:
                pass
            kids = proc.children(recursive=True)
            rss = proc.memory_info().rss + sum(_safe_rss(k) for k in kids)
            self.samples.append((psutil.cpu_percent(None), rss / 2**30, gpu))
            self._stop.wait(self.every)

    def __enter__(self):
        self.t0 = time.time(); self.th = threading.Thread(target=self._loop, daemon=True); self.th.start(); return self

    def __exit__(self, *exc):
        self._stop.set(); self.th.join(timeout=5)
        s = np.array([[a, b] for a, b, _ in self.samples]) if self.samples else np.zeros((1, 2))
        g = [x[2] for x in self.samples if x[2] is not None]
        append_jsonl(self.out, {"stage": self.stage, "wall_s": round(time.time() - self.t0, 2),
                                "cpu_pct_mean": float(s[:, 0].mean()), "rss_gb_max": float(s[:, 1].max()),
                                "gpu_gb_max": max(g) if g else None, "ok": exc[0] is None, "time": time.time()})


def _safe_rss(p):
    try:
        return p.memory_info().rss
    except Exception:
        return 0


@contextmanager
def timer(d: dict, name: str):
    t = time.perf_counter()
    yield
    d[name] = time.perf_counter() - t


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "AUSENTE"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def md_table(rows: list[dict], cols: list[str], nd=3) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(fmt(r.get(c), nd) for c in cols) + " |" for r in rows]
    return "\n".join(out)
