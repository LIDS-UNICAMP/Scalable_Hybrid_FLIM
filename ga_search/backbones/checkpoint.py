"""Checkpoint autocontido de um Bk: um único arquivo ``torch.save`` com

* ``spec_json``: o ``BackboneSpec`` completo (com ``arch_sha256``);
* ``state_dict``: encoder + cabeça (sem pai, sem decoder/projetor);
* ``state_sha256``: hash dos tensores (verificado no load);
* ``torchscript``: bytes de ``torch.jit.trace`` do modelo em eval() -> infere só com ``torch``
  (``load_standalone``), sem importar este repo.

Carregável com ``torch.load(path, weights_only=True)`` também no torch 2.2 (só str/int/tensores/dicts; TorchScript = tensor uint8).
"""
from __future__ import annotations

import hashlib
import io
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import torch

from ga_search.backbones.build import Backbone, measure_cost
from ga_search.backbones.spec import BackboneSpec

FORMAT = "backbone_ckpt/v1"


def state_sha256(sd: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        v = sd[k]
        h.update(k.encode())
        h.update(str(tuple(v.shape)).encode() + str(v.dtype).encode())
        flat = v.detach().cpu().contiguous().reshape(-1)
        h.update((flat if v.dtype == torch.bool else flat.view(torch.uint8)).numpy().tobytes())
    return h.hexdigest()


def _trace(model: Backbone, resolution: int) -> bytes:
    m = model.eval()
    x = torch.rand(2, int(model.spec.input.in_channels), resolution, resolution)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts = torch.jit.trace(m, x, check_trace=False)
    buf = io.BytesIO()
    torch.jit.save(ts, buf)
    return buf.getvalue()


def save_backbone(model: Backbone, path: Union[str, Path], *, torchscript: bool = True,
                  trace_resolution: int = 64) -> dict:
    model = model.cpu()
    was = model.training
    if not model.spec.cost.measured:
        model.spec.cost = measure_cost(model)
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    payload = {"format": FORMAT, "spec_json": model.spec.to_json(), "arch_sha256": model.spec.arch_hash(),
               "state_dict": sd, "state_sha256": state_sha256(sd), "torch_version": str(torch.__version__),
               "embed_dim": int(model.spec.head.embed_dim)}
    if torchscript:
        # uint8 tensor (não bytes): torch 2.2 com weights_only=True recusa bytes (_codecs.encode)
        payload["torchscript"] = torch.frombuffer(bytearray(_trace(model, trace_resolution)), dtype=torch.uint8).clone()
    model.train(was)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "file_sha256": file_sha, "state_sha256": payload["state_sha256"],
            "arch_sha256": payload["arch_sha256"]}


def ts_bytes(v) -> bytes:
    """TorchScript gravado como tensor uint8 (v2) ou bytes (v1)."""
    return bytes(v) if isinstance(v, (bytes, bytearray)) else v.contiguous().numpy().tobytes()


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_payload(path, map_location="cpu", expected_sha256: Optional[str] = None) -> dict:
    """``weights_only=True``; checkpoint v1 (bytes) recusado pelo torch 2.2 -> ``weights_only=False`` SÓ se o
    sha256 do arquivo for o esperado (gravado na linhagem/estado)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        if not expected_sha256 or file_sha256(path) != expected_sha256:
            raise ValueError(f"{path}: weights_only recusou ({exc}); fallback exige sha256 conferido") from exc
        return torch.load(path, map_location=map_location, weights_only=False)


def load_backbone(path: Union[str, Path], map_location="cpu",
                  expected_sha256: Optional[str] = None) -> Tuple[Backbone, BackboneSpec]:
    """Reconstrói pelo spec (precisa deste pacote) e confere arch/state hash; strict=True."""
    ck = load_payload(path, map_location, expected_sha256)
    if ck.get("format") != FORMAT:
        raise ValueError(f"formato {ck.get('format')!r} != {FORMAT}")
    spec = BackboneSpec.from_json(ck["spec_json"])
    if spec.arch_hash() != ck["arch_sha256"]:
        raise ValueError("arch_sha256 não confere")
    if state_sha256(ck["state_dict"]) != ck["state_sha256"]:
        raise ValueError("state_sha256 não confere")
    model = Backbone(spec)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.spec.cost = spec.cost
    return model.eval(), model.spec


def load_standalone(path: Union[str, Path]):
    """Só torch: devolve o módulo TorchScript (eval) e o spec em dict JSON."""
    import json
    ck = torch.load(path, map_location="cpu", weights_only=True)
    ts = torch.jit.load(io.BytesIO(ts_bytes(ck["torchscript"])), map_location="cpu")
    return ts.eval(), json.loads(ck["spec_json"])
