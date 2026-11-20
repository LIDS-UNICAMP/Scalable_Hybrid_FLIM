"""SPiFiL-0 dentro da interface (desenho §4) — só COMPÕE código existente, nada de produção é alterado.

* ``spifil0`` (S0): adaptador do init atual. ``init_block`` chama ``ga_search.modehb.init.init_block`` com os
  MESMOS argumentos que ``run_search.Evaluator._grow`` usa e devolve o MESMO objeto ``block`` — bit a bit igual
  (testado em ``tests/abstract_init``), inclusive o estado do RNG global depois da chamada.
* ``s0plus`` (S0+): o mesmo caminho, com as opções que o repositório JÁ tinha em ``flim.spifil.GrowOneLayer``
  (``whiten`` ZCA, ``patch_norm``, ``knn_smooth``, ``select_diverse``) e a dobra ZCA/escala nos pesos idêntica à de
  ``flim.spifil.grow`` (linhas ~910–922). Injetadas por troca TEMPORÁRIA de ``GrowOneLayer`` nos módulos
  ``flim.spifil`` e ``flim.progressive`` (restaurada no ``finally``). Vale para todo fit DENSO (conv/pointwise,
  ramos inception, pointwise do dwsep, projeção residual); o depthwise do SPiFiL (``fit_depthwise``) não passa
  pela estratégia e fica igual ao S0 — registrado em ``approximations``.
* ``s0data`` (S0-data): S0 com mais doadores — ``images_total`` (padrão 256, o ``max_images`` do SPiFiL-1..3)
  divididos por classe (ipc = ceil(images_total / n_classes), limitado ao pool). Controle de escassez de
  candidatos (desenho §2.2-2). O runner aplica o ipc no genoma (``genome_overrides``) para que doadores,
  logs e ``result.json`` do Evaluator usem o mesmo conjunto.

Rótulos: os três usam rótulo de imagem (Fisher, alocação por classe) e máscara quando o dataset tem
(``impurities: false``) — ``labels_used = True``.
"""
from __future__ import annotations

import contextlib
import math

import torch
from torch import nn

from ga_search.abstract_init.base import (InitContext, KernelInitializer, TrainData, check_train_only,
                                          register, split_state)


def _donors_of(train_data: TrainData):
    return list(train_data.donors)


def _augment(prov: dict, method: str, version: str, context: InitContext, train_data: TrainData, donors,
             approximations=(), extra=None) -> dict:
    cfg = dict(context.spifil_config or {})
    out = dict(prov)
    out.update({
        "abstract_init": {"method": method, "version": version, "config": dict(context.config),
                          "approximations": list(approximations)},
        "method": method, "labels_used": True,
        "label_source": "image_labels" + ("" if cfg.get("impurities", False) else "+donor_masks_if_present"),
        "mask_used": not cfg.get("impurities", False),
        "data": {"split_key": train_data.split_key, "split_fingerprint": train_data.split_fingerprint,
                 "n_images": len(donors), "n_patches": prov.get("patch_count"),
                 "patch_dim": prov.get("patch_dimension")},
    })
    if extra:
        out.update(extra)
    return out


@register("spifil0")
class Spifil0(KernelInitializer):
    """S0 = produção. ``init_block`` devolve o bloco de ``init.init_block`` sem remontar (bitwise)."""
    name, version = "spifil0", "frozen-ref"

    def _call_production(self, genome, layer_idx, prefix, donors, context: InitContext):
        from ga_search.modehb import init as I
        return _ORIGINAL_INIT_BLOCK(genome, layer_idx, prefix, donors, dict(context.spifil_config),
                                    context.prefix_id) if _ORIGINAL_INIT_BLOCK is not None else \
            I.init_block(genome, layer_idx, prefix, donors, dict(context.spifil_config), context.prefix_id)

    def _approximations(self):
        return []

    def _context(self):
        return contextlib.nullcontext()

    def genome_overrides(self, genome: dict, n_classes: int, pool_max: int) -> dict:
        return {}

    def init_block(self, genome, layer_idx, prefix, train_data: TrainData, context: InitContext):
        check_train_only(train_data)
        donors = _donors_of(train_data)
        g = {**genome, "init": "spifil"}
        with self._context() as extra:
            block, prov = self._call_production(g, layer_idx, prefix, donors, context)
        prov = _augment(prov, self.name, self.version, context, train_data, donors, self._approximations(),
                        extra=(extra if isinstance(extra, dict) else None))
        block.initialization_provenance = prov
        return block, prov

    def fit(self, prefix, train_data, block_spec, context):
        """Caminho da interface: com ``context.genome`` = equivalência com ``init.init_block``; sem genoma, o
        ``spifil_ops.fit_ops_block`` direto no ops spec já reparado (como ``_init_block_ops`` faz)."""
        check_train_only(train_data)
        donors = _donors_of(train_data)
        with self._context() as extra:
            if context.genome is not None:
                block, prov = self._call_production({**context.genome, "init": "spifil"}, context.layer_idx,
                                                    prefix, donors, context)
            else:
                from ga_search.modehb import spifil_ops as so
                cfg = {**dict(context.spifil_config), "seed": int(context.seed)}
                block, prov = so.fit_ops_block(prefix, donors, block_spec, cfg, prefix_id=context.prefix_id)
        w, b = split_state(block)
        return w, b, _augment(prov, self.name, self.version, context, train_data, donors, self._approximations(),
                              extra=(extra if isinstance(extra, dict) else None))


# ─── S0+ ──────────────────────────────────────────────────────────────────────

def check_finite_patches(learn, record: Optional[dict] = None) -> None:
    """growth_fix8: ``InfeasibleCandidate('spifil0_nonfinite_patches')`` se os patches da camada nova têm
    NaN/Inf (contados e registrados em ``record['nonfinite_patches']``). Sem efeito para entrada finita."""
    from ga_search.abstract_init.base import InfeasibleCandidate
    feats = learn.state[learn.n_layers - 1].patches.feats
    ok = torch.isfinite(feats)
    if bool(ok.all()):
        return
    bad_rows = int((~ok).any(1).sum()) if feats.dim() == 2 else int((~ok).sum())
    info = {"layer": int(learn.n_layers), "N": int(feats.shape[0]), "D": int(feats.shape[-1]),
            "nonfinite_values": int((~ok).sum()), "nonfinite_rows": bad_rows,
            "nan": int(torch.isnan(feats).sum()), "inf": int(torch.isinf(feats).sum())}
    if record is not None:
        record.setdefault("nonfinite_patches", []).append(info)
    print(f"[grow] RECUSADO (fix8): patches não finitos {info} -> InfeasibleCandidate (sem eigh/MKL)", flush=True)
    raise InfeasibleCandidate("spifil0_nonfinite_patches", details=info)


def _folded_grow_class(whiten: bool, patch_norm: bool, select_diverse: float, knn_smooth: int, record: dict):
    from flim.spifil import GrowOneLayer

    class FoldedGrowOneLayer(GrowOneLayer):
        """GrowOneLayer com as opções ligadas + a dobra ZCA/escala de ``flim.spifil.grow`` após o fit."""

        def __init__(self, budget_check: bool = True, *_, **__):
            super().__init__(budget_check, whiten, patch_norm, select_diverse, knn_smooth)

        def run(self, learn):
            # growth_fix8: patches não finitos (NaN/Inf vindos do prefixo do pai) iam direto para o
            # ``eigh(cov)`` do ZCA na CPU -> "Intel MKL ERROR: Parameter 4/5 ... DLASCL" (uma vez por
            # subproblema do divide-and-conquer: 48 linhas em D=720) e filtros NaN. Agora é recusado de forma
            # determinística ANTES da álgebra; o NewFilterInit registra o fallback (init PyTorch). Entrada
            # finita: nada muda.
            check_finite_patches(learn, record)
            super().run(learn)
            block = learn.model.block(learn.n_layers)
            with torch.no_grad():
                w = block.conv.weight
                if self.zca is not None:      # filtro no espaço branco -> W f no espaço original (W simétrica)
                    w.copy_((w.flatten(1) @ self.zca.to(w)).view_as(w))
                if self.scale is not None:    # patch_norm: P (W f) / desvio médio
                    w.copy_((w - w.mean((1, 2, 3), keepdim=True)) / self.scale)
            record["dense_fits"] = record.get("dense_fits", 0) + 1
            record.setdefault("zca_folded", 0)
            record["zca_folded"] += int(self.zca is not None)

    return FoldedGrowOneLayer


@contextlib.contextmanager
def grow_options(whiten=True, patch_norm=False, select_diverse=0.0, knn_smooth=0):
    """Troca temporária de ``GrowOneLayer`` em ``flim.spifil`` (usado por ``spifil_ops._learner``) e
    ``flim.progressive`` (``fit_new_block``). Restaura sempre. Não é thread-safe (o runner é sequencial)."""
    import flim.progressive as FP
    import flim.spifil as FS
    record: dict = {}
    cls = _folded_grow_class(bool(whiten), bool(patch_norm), float(select_diverse), int(knn_smooth), record)
    old_s, old_p = FS.GrowOneLayer, FP.GrowOneLayer
    FS.GrowOneLayer, FP.GrowOneLayer = cls, cls
    try:
        yield {"s0plus_record": record}
    finally:
        FS.GrowOneLayer, FP.GrowOneLayer = old_s, old_p


@register("s0plus")
class Spifil0Plus(Spifil0):
    name, version = "s0plus", "0.1.0"
    DEFAULTS = dict(whiten=True, patch_norm=False, select_diverse=0.9, knn_smooth=0)

    def __init__(self, **config):
        super().__init__(**{**self.DEFAULTS, **config})

    def _context(self):
        c = self.config
        return grow_options(bool(c["whiten"]), bool(c["patch_norm"]), float(c["select_diverse"]),
                            int(c["knn_smooth"]))

    def _approximations(self):
        out = ["growoneLayer_options_on_dense_fits_only(depthwise_unchanged)"]
        if self.config.get("whiten"):
            out.append("zca_folded_into_weight(bias_kept,as_flim.spifil.grow)")
        if self.config.get("patch_norm"):
            out.append("patch_norm:linear_part_folded(mean_std)")
        return out

    def _call_production(self, genome, layer_idx, prefix, donors, context):
        ctx = InitContext(**{**context.__dict__, "config": {**dict(context.config), **self.config}})
        return super()._call_production(genome, layer_idx, prefix, donors, ctx)


# ─── S0-data ─────────────────────────────────────────────────────────────────

@register("s0data")
class Spifil0Data(Spifil0):
    name, version = "s0data", "0.1.0"
    DEFAULTS = dict(images_total=256, ipc=None)

    def __init__(self, **config):
        super().__init__(**{**self.DEFAULTS, **config})

    def ipc_for(self, n_classes: int, pool_max: int | None = None) -> int:
        k = self.config.get("ipc")
        k = int(k) if k else int(math.ceil(int(self.config["images_total"]) / max(1, int(n_classes))))
        return max(1, min(k, int(pool_max))) if pool_max else max(1, k)

    def genome_overrides(self, genome: dict, n_classes: int, pool_max: int) -> dict:
        return {"spifil_images_per_class": max(int(genome.get("spifil_images_per_class", 1)),
                                               self.ipc_for(n_classes, pool_max))}

    def init_block(self, genome, layer_idx, prefix, train_data, context):
        if train_data.donor_pool:              # os doadores têm de ser os k primeiros do pool (aninhado)
            from ga_search.modehb.run_search import spifil_subset
            k = int(genome.get("spifil_images_per_class", 1))
            want = [s.name for s in spifil_subset(train_data.donor_pool, k)]
            got = [s.name for s in train_data.donors]
            if want != got:
                raise ValueError(f"s0data: doadores não são os {k} primeiros do pool por classe")
        return super().init_block(genome, layer_idx, prefix, train_data, context)


# referência à função de produção ANTES de qualquer patch do runner (o runner troca init.init_block
# pelo despachante durante um run; o S0 tem de chamar a original, não o despachante).
try:
    from ga_search.modehb.init import init_block as _ORIGINAL_INIT_BLOCK
except Exception:   # pragma: no cover
    _ORIGINAL_INIT_BLOCK = None
