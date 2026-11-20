"""Integridade do H-MODEHB (desenho secao 7): hash de config + deriva, trava de controlador,
identidade de candidato (cand_key) e adocao de resultados ja gravados no resume.

Nada aqui importa torch: tudo testavel em CPU e sem GPU.
"""
from __future__ import annotations

import datetime as _dt
import fcntl
import hashlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Iterable, Optional

# chaves fora do hash de config: nao mudam o experimento (modo do W&B, dispositivo)
DRIFT_EXCLUDE = ("wandb.mode", "device", "hmodehb.allow_drift", "growth.oom.retry_failed")  # só política de resume
ARMS = ("A", "B", "C", "D0", "D", "E")
ID_PREFIX = "h4"


# ────────────────────────────────────────────────────────────────── config hash / deriva
class ConfigDrift(RuntimeError):
    """O config atual difere do gravado no search_state.json (resume com outra receita)."""


def _plain(x):
    return json.loads(json.dumps(x, sort_keys=True, default=str))


def flatten(d, prefix: str = "") -> dict:
    out = {}
    if isinstance(d, dict):
        if not d and prefix:
            out[prefix] = {}
        for k in sorted(d, key=str):
            out.update(flatten(d[k], f"{prefix}.{k}" if prefix else str(k)))
    else:
        out[prefix] = d
    return out


def _strip(cfg: dict, exclude: Iterable[str] = DRIFT_EXCLUDE) -> dict:
    flat = flatten(_plain(cfg))
    ex = tuple(exclude)
    return {k: v for k, v in flat.items() if not any(k == e or k.startswith(e + ".") for e in ex)}


def config_sha(cfg: dict, exclude: Iterable[str] = DRIFT_EXCLUDE) -> str:
    """sha256 do config completo (search + eval + train + data + ...) canonico (chaves ordenadas)."""
    blob = json.dumps(_strip(cfg, exclude), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def config_diff(old: dict, new: dict, exclude: Iterable[str] = DRIFT_EXCLUDE) -> list:
    a, b = _strip(old, exclude), _strip(new, exclude)
    return sorted(k for k in set(a) | set(b) if a.get(k, "<ausente>") != b.get(k, "<ausente>"))


def _allowed(key: str, allow: Iterable[str]) -> bool:
    return any(key == a or key.startswith(a + ".") for a in allow)


def check_drift(old_state: dict, cfg: dict, allow: Iterable[str] = ()) -> Optional[dict]:
    """None se nao ha deriva; dict de registro se a deriva e toda permitida; ``ConfigDrift`` senao."""
    old_cfg = old_state.get("config_full")
    old_sha = old_state.get("config_sha")
    if old_cfg is None or old_sha is None:
        raise ConfigDrift("search_state.json sem config_full/config_sha: nao e um estado H-MODEHB")
    new_sha = config_sha(cfg)
    if new_sha == old_sha:
        return None
    keys = config_diff(old_cfg, cfg)
    allow = tuple(allow or ())
    bad = [k for k in keys if not _allowed(k, allow)]
    if bad:
        a, b = _strip(old_cfg), _strip(cfg)
        show = {k: [a.get(k, "<ausente>"), b.get(k, "<ausente>")] for k in bad[:20]}
        raise ConfigDrift(f"config mudou desde o inicio da busca ({len(bad)} chave(s) nao permitidas): {show}. "
                          f"Use --allow-drift <chave> so se a mudanca for intencional (fica no drift_log).")
    return {"time": time.time(), "keys": keys, "old_sha": old_sha, "new_sha": new_sha, "allowed": list(allow)}


# ────────────────────────────────────────────────────────────────── trava de controlador
class ControllerLocked(RuntimeError):
    """Outro controlador ja segura o lock deste search_state."""


class ControllerLock:
    """``fcntl.flock`` exclusivo e nao bloqueante em ``<out>/controller.lock``.

    O lock e do kernel (some quando o processo morre, inclusive kill -9): nunca fica preso. O arquivo
    guarda pid/host/inicio do dono atual so para diagnostico."""

    def __init__(self, out_dir, name: str = "controller.lock"):
        self.path = Path(out_dir) / name
        self.fd = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            try:
                owner = os.pread(fd, 4096, 0).decode(errors="replace").strip()
            finally:
                os.close(fd)
            raise ControllerLocked(f"{self.path} ja esta travado por outro controlador: {owner or '?'}")
        info = {"pid": os.getpid(), "host": socket.gethostname(),
                "start": _dt.datetime.now().isoformat(timespec="seconds")}
        os.ftruncate(fd, 0)
        os.pwrite(fd, json.dumps(info).encode(), 0)
        os.fsync(fd)
        self.fd = fd
        return self

    def release(self):
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
                self.fd = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False


# ────────────────────────────────────────────────────────────────── ids e chaves
def check_arm(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"braco {arm!r} invalido; use um de {ARMS}")
    return arm


def cand_id(arm: str, n: int) -> str:
    return f"{ID_PREFIX}-{check_arm(arm)}-c{int(n):05d}"


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()[:20]


def lineage_key(canonical_genome: dict, seed, origin: dict, fitness_sig: str) -> str:
    """Identidade do candidato sem a fidelidade: genoma efetivo + seed + origem dos pesos + regua.
    Uma promocao (mesmo candidato, fidelidade maior) mantem a lineage_key."""
    return _h({"g": canonical_genome, "seed": seed, "origin": origin, "fsig": fitness_sig})


def cand_key(lkey: str, fidelity) -> str:
    """Identidade de um JOB: lineage_key + fidelidade (= genoma + origem + fidelidade + assinatura)."""
    return _h({"l": lkey, "fid": float(fidelity)})


def file_sha256(path, chunk: int = 1 << 20) -> Optional[str]:
    p = Path(path) if path else None
    if p is None or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# ────────────────────────────────────────────────────────────────── adocao no resume
def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with tmp.open("w") as f:
        f.write(json.dumps(value, indent=1, sort_keys=True, default=str))
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def read_json(path) -> Optional[dict]:
    try:
        v = json.loads(Path(path).read_text())
    except Exception:          # ausente, parcial (kill durante a escrita) ou corrompido -> nao adota
        return None
    return v if isinstance(v, dict) else None


def adoptable(result: Optional[dict], *, ckey: str, fitness_sig: str) -> Optional[str]:
    """Motivo para NAO adotar (None = adotavel). Exige mesma cand_key e regua; se ok, checkpoint
    presente com o mesmo sha256 gravado (nunca adota checkpoint parcial/trocado)."""
    if result is None:
        return "sem resultado"
    if result.get("cand_key") != ckey:
        return f"cand_key {result.get('cand_key')} != {ckey}"
    if result.get("fitness_sig") != fitness_sig:
        return f"fitness_sig {result.get('fitness_sig')} != {fitness_sig}"
    if result.get("status", "ok") == "ok":
        ck, sha = result.get("checkpoint"), result.get("checkpoint_sha256")
        if not ck or not sha:
            return "resultado ok sem checkpoint/sha256"
        got = file_sha256(ck)
        if got != sha:
            return f"checkpoint {ck} ausente ou com sha256 diferente"
    return None

