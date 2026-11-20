"""Extracao fp32 deterministica de embeddings oficiais + mapas por bloco (hooks).

Embedding oficial = ``pooling_head.encode_tta(encoder, head, x, n=1, readout="head")``, i.e. o
mesmo ``head(encoder(x))`` de ``run_search.Evaluator._embed`` SEM TTA e SEM autocast (fp32 puro,
TF32 desligado durante a extracao). Sem head: GAP (media espacial) do ultimo mapa.
Mapas por bloco: forward hooks nos filhos de topo do encoder (``nn.Sequential`` de blocos do
MO-DEHB, os mesmos que ``CovReadout._blocks`` percorre), nomeados L1..Ln.
"""
from __future__ import annotations

import contextlib
import os
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch
import torch.nn as nn
from torch import Tensor

MapReducer = Callable[[str, Tensor], Tensor]


def official_embedding(encoder: nn.Module, head: nn.Module | None, x: Tensor) -> Tensor:
    """Embedding oficial do repo (Evaluator._embed com tta=1, readout="head"), em fp32."""
    if head is None:
        return encoder(x).float().mean(dim=(2, 3))
    from ga_search.modehb.pooling_head import encode_tta
    return encode_tta(encoder, head, x, n=1, readout="head").float()


def gap_map(_name: str, m: Tensor) -> Tensor:
    """Reducer padrao dos mapas: GAP [B,C,H,W] -> [B,C] (fp32)."""
    return m.float().mean(dim=(2, 3))


def block_modules(encoder: nn.Module) -> list[tuple[str, nn.Module]]:
    """Blocos L1..Ln = filhos de topo do encoder (como CovReadout._blocks)."""
    kids = list(encoder.children())
    if not kids:
        raise ValueError("encoder sem filhos: nao ha blocos para L1..Ln")
    return [(f"L{i + 1}", m) for i, m in enumerate(kids)]


@contextlib.contextmanager
def fp32_strict():
    """Desliga TF32 (matmul + cudnn) e restaura ao sair; nao usa autocast."""
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
           torch.get_float32_matmul_precision())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old[0], old[1]
        torch.set_float32_matmul_precision(old[2])


class MemMeter:
    """Pico de memoria. CUDA: torch.cuda.max_memory_allocated. CPU: pico do RSS (amostrado a cada
    5 ms por uma thread) menos o RSS inicial -> ``peak_bytes``; o pico do tracemalloc (so
    alocacoes Python, nao ve os buffers do torch) vai em ``tracemalloc_peak``."""

    def __init__(self, device: torch.device):
        self.device = torch.device(device)
        self.peak_bytes, self.kind, self.rss_bytes, self.tracemalloc_peak = 0, "", 0, 0

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self.kind = "cuda_max_memory_allocated"
            return self
        import threading
        self._started = not tracemalloc.is_tracing()
        if self._started:
            tracemalloc.start()
        tracemalloc.reset_peak()
        self.kind = "cpu_rss_peak_delta"
        self._rss0 = self._rss_max = _rss_bytes()
        self._stop = threading.Event()

        def sample():
            while not self._stop.wait(0.005):
                self._rss_max = max(self._rss_max, _rss_bytes())
        self._thr = threading.Thread(target=sample, daemon=True)
        self._thr.start()
        return self

    def __exit__(self, *exc):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self.peak_bytes = int(torch.cuda.max_memory_allocated(self.device))
        else:
            self._stop.set()
            self._thr.join()
            self._rss_max = max(self._rss_max, _rss_bytes())
            self.peak_bytes = int(self._rss_max - self._rss0)
            self.tracemalloc_peak = int(tracemalloc.get_traced_memory()[1])
            if self._started:
                tracemalloc.stop()
        self.rss_bytes = _rss_bytes()
        return False


def _rss_bytes() -> int:
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return 0


@dataclass
class Extraction:
    embedding: Tensor                       # [N, D] oficial (head(encoder(x)) ou GAP)
    labels: Tensor                          # [N]
    ids: list = field(default_factory=list)
    layers: dict = field(default_factory=dict)   # nome -> reducer(mapa) concatenado [N, ...]
    maps: dict = field(default_factory=dict)     # nome -> mapas crus [<=max_map_images, C, H, W]
    images: Tensor | None = None            # entradas (se keep_images)
    time_s: float = 0.0
    peak_mem_bytes: int = 0                 # CUDA: max_memory_allocated; CPU: pico RSS - RSS inicial
    mem_kind: str = ""
    tracemalloc_peak: int = 0
    rss_bytes: int = 0
    device: str = "cpu"
    batch_size: int = 0


def _batches(data, batch_size: int) -> Iterable[tuple]:
    """Aceita DataLoader (x, y[, ids]) ou tupla (X, y[, ids]) de tensores em RAM."""
    if isinstance(data, (tuple, list)) and len(data) >= 2 and isinstance(data[0], Tensor):
        X, y = data[0], torch.as_tensor(data[1])
        ids = list(data[2]) if len(data) > 2 else list(range(len(X)))
        for i in range(0, len(X), batch_size):
            yield X[i:i + batch_size], y[i:i + batch_size], ids[i:i + batch_size]
        return
    for b in data:
        x, y = b[0], torch.as_tensor(b[1])
        ids = list(b[2]) if len(b) > 2 else [None] * len(x)
        for i in range(0, len(x), batch_size):     # re-lote em lotes pequenos
            yield x[i:i + batch_size], y[i:i + batch_size], ids[i:i + batch_size]


@torch.no_grad()
def extract(encoder: nn.Module, head: nn.Module | None, data, *, device="cpu", batch_size: int = 8,
            layers: bool = True, map_reducer: MapReducer | None = gap_map, max_map_images: int = 0,
            keep_images: bool = False, transform: Callable[[Tensor], Tensor] | None = None) -> Extraction:
    """Extrai embedding oficial (+ mapas por bloco) em eval/no_grad/fp32.

    Args:
        data: DataLoader que rende (x, y[, ids]) ou tupla (X, y[, ids]).
        batch_size: lote maximo por forward (re-lota os lotes do loader).
        layers: registra hooks em L1..Ln; ``map_reducer(nome, mapa)`` reduz cada lote
            (padrao GAP) e vai para ``Extraction.layers``.
        max_map_images: guarda os mapas crus dos primeiros N exemplos (0 = nenhum).
        transform: perturbacao aplicada a x (ja normalizado) antes do forward (textura).
    """
    device = torch.device(device)
    encoder = encoder.to(device).float().eval()
    if head is not None:
        head = head.to(device).float().eval()
    cur: dict[str, Tensor] = {}
    hooks = []
    if layers:
        for name, mod in block_modules(encoder):
            hooks.append(mod.register_forward_hook(
                lambda _m, _i, out, name=name: cur.__setitem__(name, out)))
    embs, ys, ids, imgs = [], [], [], []
    red: dict[str, list] = {}
    raw: dict[str, list] = {}
    n_raw = 0
    t0 = time.perf_counter()
    try:
        with fp32_strict(), MemMeter(device) as mm:
            for x, y, bid in _batches(data, batch_size):
                x = x.to(device).float()
                if transform is not None:
                    x = transform(x)
                cur.clear()
                embs.append(official_embedding(encoder, head, x).cpu())
                ys.append(y.cpu()), ids.extend(bid)
                if keep_images:
                    imgs.append(x.cpu())
                if layers:
                    take = max(0, min(len(x), max_map_images - n_raw))
                    for name, m in cur.items():
                        if map_reducer is not None:
                            red.setdefault(name, []).append(map_reducer(name, m).float().cpu())
                        if take:
                            raw.setdefault(name, []).append(m[:take].float().cpu())
                    n_raw += take
            if device.type == "cuda":
                torch.cuda.synchronize(device)
    finally:
        for h in hooks:
            h.remove()
    dt = time.perf_counter() - t0
    return Extraction(embedding=torch.cat(embs), labels=torch.cat(ys), ids=ids,
                      layers={k: torch.cat(v) for k, v in red.items()},
                      maps={k: torch.cat(v) for k, v in raw.items()},
                      images=torch.cat(imgs) if keep_images else None, time_s=dt,
                      peak_mem_bytes=mm.peak_bytes, mem_kind=mm.kind, tracemalloc_peak=mm.tracemalloc_peak, rss_bytes=mm.rss_bytes,
                      device=str(device), batch_size=int(batch_size))


def load_images(data) -> tuple[Tensor, Tensor, list]:
    """Carrega (X, y, ids) do loader em RAM (como svm_on_demand._cov_features)."""
    xs, ys, ids = [], [], []
    for b in _batches(data, 10 ** 9):
        xs.append(b[0].float().cpu()), ys.append(b[1]), ids.extend(b[2])
    return torch.cat(xs), torch.cat(ys), ids


def load_checkpoint(path, map_location="cpu") -> tuple[nn.Module, nn.Module | None, dict]:
    """Le um checkpoint SEM alterar pesos: {"encoder", "head", ...} (MO-DEHB encoder.pt) ou nn.Module."""
    ck = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(ck, nn.Module):
        return ck, None, {}
    if isinstance(ck, dict) and isinstance(ck.get("encoder"), nn.Module):
        meta = {k: v for k, v in ck.items() if k not in ("encoder", "head") and isinstance(v, (int, float, str))}
        return ck["encoder"], ck.get("head"), meta
    raise ValueError(f"{path}: formato de checkpoint nao suportado (esperado dict com 'encoder' nn.Module)")
