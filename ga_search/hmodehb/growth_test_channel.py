"""Canal de TESTE da cadeia de crescimento (R2/R7: SVM original no teste de todo aceito; nunca decide nada).

Roda num processo SEPARADO do controlador (``growth_chain`` só o dispara e nunca lê a saída). Lê
``<out>/test_queue.jsonl`` (aceitos), monta o MESMO treino do SVM da fitness (régua ORIG: ``head(encoder(x))``,
``core.svm.fit_svm`` linear, C=100, sem scaler, via ``supervised_fitness.svm_fitness``) e mede o kappa no split
de teste. Cada linha de ``<out>/test_orig.jsonl`` leva ``never_feeds_fitness=True`` e
``used_for_selection=False``. Idempotente: (node_id, sha256 do checkpoint) já medido é pulado.

    python -m ga_search.hmodehb.growth_test_channel --out artifacts/hmodehb_b0b4/growth/<ds>/seed42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

TEST_FILE = "test_orig.jsonl"


def _done(path: Path) -> set:
    out = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                out.add((r["node_id"], r["checkpoint_sha256"]))
            except Exception:
                pass
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    q = out / "test_queue.jsonl"
    if not q.exists():
        return 0
    st = json.loads((out / "growth_state.json").read_text())
    cfg, g = st["config_full"]["run"], st["config_full"]["growth"]
    todo, seen = [], _done(out / TEST_FILE)
    for line in q.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        k = (r["node_id"], r["checkpoint_sha256"])
        if k not in seen:
            seen.add(k)
            todo.append(r)
    if not todo:
        return 0
    if (cfg.get("data") or {}).get("synthetic"):
        print("[test_channel] dados sintéticos: sem split de teste", flush=True)
        return 0
    from ga_search.backbones.checkpoint import load_backbone
    from ga_search.hmodehb.growth_chain import GrowthEvaluator, fp32_reading
    from ga_search.hmodehb.integrity import file_sha256
    from ga_search.modehb.supervised_fitness import svm_fitness
    from ga_search.modehb.svm_on_demand import _test_loader
    gev = GrowthEvaluator(cfg, out / "eval_test_channel", g.get("debug_train"), g.get("debug_val"), v3=g.get("v3"))
    v3 = (g.get("v3") or {})
    ev = gev.ev
    xs, ys = [], []
    for x, y, *_ in _test_loader(cfg["data"]):
        xs.append(x), ys.append(torch.as_tensor(y))
    tx, ty = torch.cat(xs), torch.cat(ys).numpy()
    for r in todo:
        if file_sha256(r["checkpoint"]) != r["checkpoint_sha256"]:
            continue
        model, spec = load_backbone(r["checkpoint"], expected_sha256=r["checkpoint_sha256"])
        model.to(ev.device)
        enc = nn.Sequential(*model.blocks)
        svm = lambda a, b: svm_fitness(a, ev.svm_train_y.numpy(), b, ty, C=100.0, scaler="none", whiten="none",  # noqa
                                       kappa_weight=1.0, max_train=ev.sv2["max_train"], seed=ev.seed)
        extra = {}
        if gev.readout != "head":            # growth v3 (1)(5): cabeça de 2ª ordem (PCA só no treino) + TTA8 + GAP aux
            from ga_search.hmodehb.so2_head import So2Head
            saved = Path(r["checkpoint"]).parent / "so2_head.pt"
            sh = So2Head.from_payload(torch.load(saved, weights_only=False)) if saved.exists() else None
            # a cabeça SALVA com o modelo (a que a seleção usou); sem arquivo: re-ajuste no treino (registrado)
            head, (tr, te), gaps = gev.so2_read(model, [tx], tta=1, gap=True, head=sh)
            sv = svm(tr.numpy(), te.numpy())
            extra.update(readout=gev.readout, so2_head_sha256=head.sha256(),
                         so2_head_source="saved (so2_head.pt)" if sh is not None else "refit_train")
            if sh is not None:                   # diagnóstico: re-ajuste neste processo (só arredondamento)
                rh, _, _ = gev.so2_read(model, [], tta=1)
                extra["so2_refit_max_abs_diff_V"] = float((rh.V - sh.V).abs().max()) if rh.V.shape == sh.V.shape else None
            sv_g = svm(gaps[0].numpy(), gaps[1].numpy())
            extra.update(kappa_test_gap=float(sv_g["kappa"]), macro_f1_test_gap=float(sv_g["macro_f1"]))
            if v3.get("test_tta8", False):
                _h8, (tr8, te8), _ = gev.so2_read(model, [tx], tta=8, head=sh)
                sv8 = svm(tr8.numpy(), te8.numpy())
                extra.update(kappa_test_tta8=float(sv8["kappa"]), macro_f1_test_tta8=float(sv8["macro_f1"]),
                             tta8="média de z sobre as 8 vistas rot/flip (treino e teste; PCA da vista identidade)")
            te = te.numpy()
        else:
            with fp32_reading(ev):               # growth_fix4: mesma leitura fp32 do controlador
                tr = ev._embed(enc, model.head, ev.svm_train_x, tta=1, readout="head").numpy()
                te = ev._embed(enc, model.head, tx, tta=1, readout="head").numpy()
            sv = svm(tr, te)
            if v3.get("test_tta8", False):
                with fp32_reading(ev):
                    tr8 = ev._embed(enc, model.head, ev.svm_train_x, tta=8, readout="head").numpy()
                    te8 = ev._embed(enc, model.head, tx, tta=8, readout="head").numpy()
                sv8 = svm(tr8, te8)
                extra.update(kappa_test_tta8=float(sv8["kappa"]), macro_f1_test_tta8=float(sv8["macro_f1"]))
        rec = {"node_id": r["node_id"], "checkpoint": r["checkpoint"], "checkpoint_sha256": r["checkpoint_sha256"],
               "band": r["band"], "fid": r["fid"], "regua": "ORIG_full" if len(ev.svm_train_y) < 4000 else "ORIG_sub4000",
               "svm_protocol": "scaler=none;kernel=linear;C=100;ovo;max_iter=-1", "split_eval": "test",
               "kappa_test": float(sv["kappa"]), "macro_f1_test": float(sv["macro_f1"]), "n_test": int(len(ty)),
               "embed_dim": int(te.shape[1]), "embed_precision": gev.embed_precision, "params": int(spec.cost.params_total), "flops_2mac": int(spec.cost.flops),
               "never_feeds_fitness": True, "used_for_selection": False, "time": time.time(), **extra}
        with (out / TEST_FILE).open("a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print(f"[test_channel] {r['node_id']} {r['band']} kappa_test={rec['kappa_test']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
