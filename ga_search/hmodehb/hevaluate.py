"""Avaliador do H-MODEHB: subclasse de ``run_search.Evaluator`` com a fitness v2.

Reusa todo o pipeline historico (SPiFiL -> AE local -> SSL global LeJEPA+SIGReg -> WiSE -> perfil de
custo -> checkpoint). Muda so a leitura da fitness:

* ``_svm_score`` chama ``supervised_fitness.svm_fitness`` com a regua ORIG fixa (C=100, scaler none,
  whiten none, kappa_weight 1.0) sobre ``head(encoder(x))``, recusando embedding < 1280;
* o resultado vira v2 (``fitness_v2.finalize``): quality = kappa ORIG sem multiplicador de colapso,
  diagnosticos em ``aux``; embedding/loss nao finitos -> ``NonFiniteFitness`` (controller: reason=nan);
* ``result.json`` leva ``cand_key``, ``fitness_sig``, ``config_sha`` e ``checkpoint_sha256`` (adocao
  no resume).

O split de teste nunca e carregado (``progressive_data.load_data`` devolve so treino/validacao).
"""
from __future__ import annotations

from ga_search.hmodehb import bk as BK
from ga_search.hmodehb import fitness_v2 as FV
from ga_search.hmodehb import integrity as I
from ga_search.modehb import run_search as RS


class EmbedDimError(FV.FitnessConfigError):
    """Cabeca com embedding < 1280 (proibido pela decisao do usuario de 17/11)."""


class HEvaluator(RS.Evaluator):
    def __init__(self, cfg, out_dir):
        self.sv2 = FV.validate_eval(cfg.get("eval") or {})
        if (cfg.get("data") or {}).get("val_max_samples") is not None:
            raise FV.FitnessConfigError("fitness v2 usa a validacao INTEIRA: data.val_max_samples deve ser null")
        super().__init__(cfg, out_dir)
        self.fitness_spec = FV.spec(self.sv2, val_sha256=self.split_hash, n_val=int(len(self.val_y)),
                                    dataset=self.dataset, split=(cfg.get("data") or {}).get("split"))
        self.fitness_sig = FV.signature(self.fitness_spec)
        self._fit_sv, self._readouts, self._svm_stamps = None, [], []

    # ---- regua ORIG (em TODA leitura que decide: alpha do WiSE e kappa final)
    def _svm_features(self, encoder, head):
        """Sempre head(encoder(x)), tta 1: nunca cov_head, qualquer que seja o config."""
        self.__dict__.setdefault("_readouts", []).append("head")
        tr = self._embed(encoder, head, self.svm_train_x, tta=1, readout="head").numpy()
        va = self._embed(encoder, head, self.val_x, tta=1, readout="head").numpy()
        return tr, va

    def _embed_cov(self, encoder, head, extra=None):
        self.__dict__.setdefault("_readouts", []).append("cov_head")      # so diagnostico; se alimentar uma decisao, o carimbo acusa
        return super()._embed_cov(encoder, head, extra)

    def _svm_score(self, tr, va):
        from ga_search.modehb.supervised_fitness import svm_fitness
        d = int(va.reshape(len(va), -1).shape[1])
        if d < BK.MIN_EMBED_DIM:
            raise EmbedDimError(f"embedding de dimensao {d} < {BK.MIN_EMBED_DIM}")
        kw = {"C": FV.REQUIRED["C"], "scaler": "none", "whiten": "none", "kappa_weight": 1.0,
              "max_train": self.sv2["max_train"]}
        if self.sv2.get("gram_reduce"):
            kw["gram_reduce"] = True
        self.__dict__.setdefault("_svm_stamps", []).append({k: kw[k] for k in ("C", "scaler", "whiten", "kappa_weight", "max_train")})
        if getattr(self, "return_pred", False):     # growth v3.1: bootstrap da aceitação (SVM inalterada)
            kw["return_pred"] = True
        return svm_fitness(tr, self.svm_train_y.numpy(), va, self.val_y.numpy(), seed=self.seed, **kw)

    def _svm_quality(self, encoder, head):
        sv = super()._svm_quality(encoder, head)
        self._fit_sv = sv
        return sv

    def _wise_select(self, encoder, head, anchor, trained, alphas, run):
        out = super()._wise_select(encoder, head, anchor, trained, alphas, run)
        self._fit_sv = out["sv"]
        return out

    # ---- uma avaliacao
    def _evaluate(self, genome, fidelity, ctx, cand_dir, run):
        if int(genome.get("embed_dim", 0)) < BK.MIN_EMBED_DIM:
            raise EmbedDimError(f"genoma com embed_dim={genome.get('embed_dim')} < {BK.MIN_EMBED_DIM}")
        if ctx.get("fitness_sig") not in (None, self.fitness_sig):
            raise FV.FitnessSigMismatch(f"controller {ctx.get('fitness_sig')} != avaliador {self.fitness_sig}")
        self._fit_sv, self._readouts, self._svm_stamps = None, [], []
        base = super()._evaluate(genome, fidelity, ctx, cand_dir, run)
        sv = self._fit_sv or {}
        res = FV.finalize(base, kappa=base.get("fitness_kappa"), nonfinite=bool(sv.get("nonfinite")))
        # carimbo de cada leitura que decidiu (readouts x kwargs do SVM; solver/tta fixos no _svm_features)
        res["decision_rulers"] = sorted({FV.ruler_sig({"readout": r, "tta": 1, "solver": "libsvm", **st})
                                         for r in set(self._readouts) for st in self._svm_stamps})
        res["n_svm_decisions"] = len(self._svm_stamps)
        res.update(cand_key=ctx.get("cand_key"), lineage_key=ctx.get("lineage_key"),
                   fitness_sig=self.fitness_sig, config_sha=ctx.get("config_sha"),
                   checkpoint_sha256=I.file_sha256(res.get("checkpoint")),
                   embed_dim=int(genome["embed_dim"]), bk=BK.bk_of(res.get("params")),
                   params_counted="encoder+head (decoders/projetores fora)")
        return res
