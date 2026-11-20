"""Cadeia de crescimento progressivo B0 -> B1 -> B2 -> B3 -> B4 (integração A2 + A3 + A4).

Prompt do usuário: "crescer devagar, sem estragar os pesos SPiFiL; cada faixa melhor que a anterior".

Pipeline de UM filho (pai = membro aceito da faixa B_k, alvo = faixa B_{k+1}):

1. morfismo que preserva a função (A3, ``ga_search.backbones.morphisms``): ``widen`` exato (+``head_widen``
   quando alarga a última camada; a cabeça >= 1280 é herdada com colunas zero e nunca encolhe),
   ``deepen_identity`` (só onde a entrada é >= 0), ``deepen_residual_zero`` e ``add_skip_zero``;
2. canais NOVOS inicializados por SPiFiL (``KernelInitializer`` do A2: ``spifil0`` padrão, ``spifil1``/``spifil2``
   por flag) ajustado sobre o prefixo HERDADO; os pesos de SAÍDA desses canais (colunas do consumidor)
   começam em zero, então a função do pai é preservada (medida em eval(), fp32, imagens de VALIDAÇÃO);
3. pesos herdados protegidos (``protect`` = lowlr padrão | freeze | full) por elemento
   (``external_seed.ElementProtect``); elementos novos com lr cheio;
4. AE local SÓ nos blocos com elementos novos -> LeJEPA+SIGReg curto -> WiSE com âncora = o filho no passo 0
   (= o PAI em função: alpha=0 devolve exatamente os pesos do morfismo) -> alpha escolhido pela ORIG-val;
5. aceitação (growth_fix4): o pai é re-pontuado no MESMO processo e na MESMA leitura fp32 do filho;
   kappa_ORIG_val(filho) >= kappa_ORIG_val(pai re-pontuado) + delta (delta=0 padrão). Preservação exata +
   alpha=0 => ``adopted_as_parent`` (o filho é o pai: não ocupa a faixa nem é promovido).

Seleção por faixa (MO-DEHB de crescimento): pais = frente de Pareto (kappa ORIG-val máx, FLOPs 2·MAC mín)
da faixa; filhos com vetor de plano theta em [0,1]^4 gerado por DE rand1/bin entre os thetas aceitos da
faixa; triagem em ``fid_low`` (f15) e promoção (1/eta melhores aceitos) em ``fid_high`` (f45), onde a
aceitação final é decidida. Teto duro 5,5M em todo filho.

Teste: o kappa ORIG de TESTE de todo aceito é calculado num SUBPROCESSO (``growth_test_channel``), gravado em
um arquivo próprio com ``never_feeds_fitness=True``; este módulo nunca abre esse arquivo.

Robustez: trava flock (A4), config_sha + deriva (A4), cand_key com sha256 do checkpoint do pai, adoção do
``result.json`` no resume (mesma cand_key/fitness_sig/sha), OOM 10 x 300 s, retomada idempotente
(plano determinístico por (faixa, geração, índice)).

    python -m ga_search.hmodehb.growth_chain --dataset remote_sensing --device cuda [--set k=v ...]
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch import nn

from ga_search.backbones import build as BB
from ga_search.backbones import morphisms as MO
from ga_search.backbones.spec import (BUDGETS, HARD_CAP, BackboneSpec, HeadSpec, InputSpec, LayerSpec,
                                      StageSpec, band_of)
from ga_search.hmodehb import fitness_v2 as FV
from ga_search.hmodehb import integrity as I

BANDS = ("B0", "B1", "B2", "B3", "B4")
GROWTH_CONFIG = Path(__file__).resolve().parents[1] / "config" / "hmodehb_b0b4" / "growth.yaml"
PROTECT_SCALE = {"full": 1.0, "lowlr": 0.01, "freeze": 0.0}
KIND = "h4_growth"
# growth_fix4 (a): toda leitura SVM do growth (sementes, pai re-pontuado, alphas do WiSE, canal de teste) usa a
# MESMA precisão: fp32, autocast desligado e TF32 desligado (cuDNN/matmul). Entra na fitness_spec -> muda a
# fitness_sig: estados/resultados lidos em FP16 (autocast sem dtype) não são adotados nem misturados.
EMBED_PRECISION = "fp32_noautocast_notf32"

GROWTH_DEFAULTS: Dict[str, Any] = {
    "n_parents": 4, "n_children": 8, "generations": 2, "eta": 3,
    "fid_low": 15, "fid_high": 45, "delta": 0.0, "require_improvement": False,
    "protect": "lowlr", "init_method": "spifil0", "init_config": {}, "spifil_images_per_class": 4,
    "start_band": "B0", "end_band": "B4", "hard_cap": HARD_CAP,
    "ops_weights": {"widen": 0.55, "deepen_residual_zero": 0.2, "deepen_identity": 0.1, "add_skip_zero": 0.15},
    "max_ops": 12, "residual_act": "silu",
    "seeds": {"registry": "research/spifil_b0b4/agent1/seed_registry.csv", "regua": "ORIG", "bands": ["B0"],
              "checkpoints": [], "external_small": True, "top_k": 4},
    "ssl": "lejepa", "reg": "sigreg", "ae_target": "pixel", "ae_schedule": "joint",
    "lr": 1.0e-3, "lr_ae": 1.0e-3, "wise_alphas": [0.0, 0.25, 0.5, 0.75, 1.0],
    # growth_v31_hotfix_ae_diverge: AE local nos blocos alargados divergia (RS: 0,63 -> 1,85; tiny 86 -> 370).
    # Causa medida (CPU, RS c00306 widen->455 e tiny c00001 widens->500): o DECODER aleatório dá o 1º passo do Adam
    # sobre z de RMS ~8-16 com fan-in 9*C (C ~ 450-500): |dw| ~ lr por peso, coerente em sinal -> a reconstrução
    # anda ~lr*9C*|z| -> pico no passo 2 (tiny 21 -> 6017; RS 0,40 -> 2,57). Não é a init SPiFiL (o W&B mostra o
    # pico com o SPiFiL congelado). Correção (só no decoder descartável; o bloco, o alvo e o lr do bloco não mudam):
    #   ae_dec_in_norm: entrada do decoder padronizada (média/desvio FIXOS do lote-sonda) -> tira |z|
    #   ae_lr_scale_fanin: lr dos pesos do decoder * sqrt(144/fan_in) -> tira o fan-in 9C
    #   (os dois juntos = padrão; só a norma ainda deixa pico 0,38 -> 4,1 no tiny; warmup/clip testados, piores)
    #   ae_grad_clip / ae_warmup_steps: opções (desligadas)
    #   ae_diverge_guard: razão + tendência por época (nunca o valor absoluto): média/inicial > guard E subindo,
    #     ou no fim último-quarto/inicial > guard, ou não finita -> o bloco volta ao estado pré-AE (o morfismo) e o
    #     filho segue sem AE nesse bloco, registrado "ae_diverged" ("começa alto e cai" passa). null = desligada
    #   ae_smoke_strict: recon_ratio > 1 em algum bloco -> o filho FALHA (ChainError). false (padrão) = só alerta
    #     ("[ae] ALERTA" + ae_alerts/ae_diverged_blocks no result.json); null = estrito se fid_high <= 1.
    #     Smoke da v3: --gset ae_smoke_strict=true
    "ae_dec_in_norm": True, "ae_grad_clip": None, "ae_lr_scale_fanin": True, "ae_warmup_steps": 0,
    "ae_diverge_guard": 1.0, "ae_smoke_strict": False,
    "preservation_images": 8, "preservation_tol": 1.0e-5,
    # oom.retry_failed: no resume, re-roda (em vez de adotar) nó failed com reason "oom" (só contenção)
    "oom": {"retries": 10, "sleep_s": 300, "retry_failed": False},
    "test_channel": True, "debug_train": None, "debug_val": None,
    # memória (growth_mem). Ligados por padrão SÓ os que não mudam a ciência (bit a bit; ver mem_status.md):
    #   grad_checkpoint: recomputa a ativação de cada bloco no backward do SSL (BN/DBN com momentum 0 na
    #     recomputação -> running stats atualizadas uma vez só; RNG preservado) -> pico = 1 bloco, não a rede.
    #   wise_on_cpu: âncora/treinado do WiSE em CPU; empty_cache: libera o cache entre fases.
    # Desligados e DECLARADOS (mudam a ciência ou podem mudar o kappa em arredondamento):
    #   embed_batch_size: micro-lote da extração da SVM (null = data.batch_size, histórico);
    #   ssl_amp_bf16: SSL em bf16 (muda a ciência).
    "mem": {"grad_checkpoint": True, "wise_on_cpu": True, "empty_cache": True,
            "embed_batch_size": None, "ssl_amp_bf16": False,
            # growth_fix5: teto do pico ESTIMADO por plano (GB); null = memória total da GPU - 3 GB (torch.cuda),
            # sem GPU = filtro desligado. Plano acima -> "infeasible_mem" e o planner gera outro.
            "max_gpu_gb": None, "oom_capacity_margin_gib": 1.0},
    # growth_fix6: promoção fid_low -> fid_high. "resume" (padrão) RETOMA o filho de f_low: pesos treinados
    # pré-WiSE + estado do AdamW/RNG (se salvos; senão AdamW novo, registrado) e treina mais
    # (fid_high - fid_low) épocas de SSL ("fid_diff"; "ssl_diff" = ssl(f_high) - ssl(f_low)), mesmo lr fixo.
    # Âncora do WiSE = o morfismo (pai em função). Nó final = melhor de f_low/f_high pela ORIG-val.
    # "retrain" = histórico (refaz do morfismo). save_optimizer: grava o AdamW em f_low.
    "promotion": {"mode": "resume", "save_optimizer": True, "extra_epochs": "fid_diff"},
    # growth v3 (cadeia de crescimento v3). PADRÕES = histórico (readout head, sem genes, sem ops novas): sem
    # deriva para quem não liga. O exemplo ligado está em config/hmodehb_b0b4/growth_v3.yaml.
    #   readout: "head" (histórico) | "so2_4096_pca" (cabeça de 2ª ordem fixa: fitness E teste; muda a fitness_sig)
    #   aux_gap: com so2, o kappa da cabeça antiga (GAP/proj) vai para aux (nunca decide)
    #   genes: listas -> cada filho recebe uma combinação determinística (produto cartesiano, rodízio por geração):
    #     ssl {lejepa, simclr, vicreg, none}; aug {null=histórico, weak, medium, strong};
    #     whiten_new {none, zca, dbn} (norm do bloco novo do deepen_residual_zero); softhebb {false, true}
    #   ops: famílias novas habilitadas {inception_branch_add, dense_concat, replace_bad_filters}
    #   families: null = todas (históricas + ops); lista = só esses tipos (braços por op)
    #   freeze_head_in_ssl: a projeção antiga da cabeça fica congelada no SSL (a so2 não tem parâmetros)
    #   saturation.adopt_parent: faixa sem filho aceito (ou pior que a anterior) -> o melhor membro da faixa
    #     anterior é ADOTADO como membro da próxima (curva kappa x faixa nunca piora; a cadeia continua)
    #   test_tta8: canal de teste com coluna extra TTA8 (nunca na seleção)
    "v3": {"readout": "head", "so2": {"dim": 4096, "n_px": 64}, "aux_gap": True,
           "genes": {"ssl": None, "aug": None, "whiten_new": None, "softhebb": None},
           "ops": [], "families": None, "branch_kernels": [1, 3, 5], "branch_frac": 0.25,
           "replace": {"corr_thr": 0.95, "var_rel_eps": 1.0e-6, "max_frac": 0.25, "n_images": 64},
           "softhebb_cfg": {"epochs": 1, "lr": 0.01, "temperature": 1.0, "n_images": 128},
           "freeze_head_in_ssl": False, "saturation": {"adopt_parent": False}, "test_tta8": False,
           # spifil_strict: filtro novo sem SPiFiL (conv não densa, exceção, SPiFiL com menos filtros) -> o filho
           #   falha como "spifil_fallback" (infeasible) em vez de receber init aleatória
           # control_no_grow: controle LEMONADE (pai treinado as mesmas épocas, SEM crescer; só registro)
           "spifil_strict": False, "control_no_grow": False, "ssl_recipe": None, "recipe_small_views": False,
           "weight_decay": None,
           # v3.1 cov_gate: ops "cov_gate" (y = x + g*act(ZCA/DBN(conv_SPiFiL(x))), g=0) e "deepen_rezero";
           # cov_gate_head: no SSL o embedding vira [cabeça antiga congelada, g_h * Cov 2ª ordem diferenciável]
           "cov_gate_mode": "channel", "cov_gate_fit_images": 64, "cov_gate_head": False, "cov_gate_head_k": 16,
           # v3.1 bloco 2 (receita fiel com batch maior): ssl_batch_size = batch do SSL (com ssl_recipe: paper);
           #   OOM -> metade, até ssl_batch_min (registrado "adaptação: batch X por memória"); época = passada
           #   completa; LARS (SimCLR/VICReg) / AdamW (LeJEPA) + warmup + cosseno pelo passo global
           "ssl_batch_size": None, "ssl_batch_min": 32,
           # v3.1 bloco 4: zca_refit = re-ajusta a ZCA dos blocos novos depois do AE/SSL (whitening.md);
           #   softhebb_paper = tau/lr por camada (1/tau = 1, .65, .25; lr .08, .005, .01), BN sem treino na entrada,
           #   filtros SPiFiL levados ao raio R (3) durante o SoftHebb e devolvidos à norma original depois
           "zca_refit": False, "zca_refit_images": 256, "softhebb_paper": False, "softhebb_radius": 3.0,
           # v3.1 bloco 5: replace.criterion = corr (v3) | redo (tau .1) | repr (p .30); prune_redundant_new = poda
           #   exata dos canais NOVOS mortos/duplicados (mais superpixels -> redundância eliminada na rede)
           "prune_redundant_new": False, "prune": {"corr_thr": 0.95, "var_rel_eps": 1.0e-6, "n_images": 64},
           # v3.1 bloco 6: full_epoch_ae (época do AE = passada completa); accept_bootstrap {n, alpha} (aceita só se o
           #   quantil alpha do Δκ pareado da val >= 0); spifil_dwsep (SPiFiL no pointwise novo do dwsep)
           "full_epoch_ae": False, "accept_bootstrap": None, "spifil_dwsep": False},
}

V3_SSL_REG = {"lejepa": "sigreg", "simclr": "none", "vicreg": "none", "barlow": "none", "byol": "none", "dino": "none"}

# growth v3: receitas dos ARTIGOS (research/spifil_b0b4/articles/*.md). ``v3.ssl_recipe: paper`` passa ao trainer o que
# ele suporta ("fiel"); o resto fica registrado como "adaptação" (inevitável: memória/dataset/LAB) ou "pendente"
# (não implementado no trainer ainda) no result.json -> nada é escondido.
V3_RECIPES = {
    "lejepa": {"kwargs": {"n_views": 8, "sigreg_lambda": 0.05, "weight_decay": 5e-4, "projector": True,
                          "projector_dims": [2048, 2048, 128], "reg_kwargs": {"n_proj": 1024}},
               # v3.1 (bloco 2): AdamW lr 5e-4, warmup linear de 1 passada (fator inicial 0,01) + cosseno até lr/1000
               "optim": {"optimizer": "adamw", "lr": 5e-4, "lr_rule": "fixo (5e-4, não escala com B)",
                         "warmup": {"epochs": 1}, "start_ratio": 0.01, "end_ratio": 1e-3, "wd_exclude_1d": False},
               "fiel": ["projetor MLP 3 camadas BN d->2048->2048->128, perda NO projetor, SVM no embedding",
                        "V=8, lambda=0,05", "AdamW wd 5e-4", "SIGReg com 1024 slices", "épocas = passadas completas"],
               "adaptacao": ["aug oficial (ColorJitter .4/.4/.2/.1 p.8, gray .2, blur .5, solarize .2) em LAB -> preset "
                             "'strong' do make_views (jitter aditivo LAB, blur, flips/rot90; sem solarize/gray)",
                             "batch = data.batch_size do braço (>=128 quando cabe; senão 'adaptação: batch X por memória')"],
               "pendente": ["warmup + cosseno até lr/1000 (trainer só tem lr fixo)"],
               "small_views": {"n_views": 4, "sigreg_lambda": 0.02}},
    "simclr": {"kwargs": {"tau": 0.5, "weight_decay": 1e-4, "projector_dims": [0, 128]},
               # v3.1: LARS (eta 1e-3, m 0,9; bias/BN fora do wd e da adaptação), lr = 0,075*sqrt(B) (v1 Tab. B.1,
               # batch pequeno), warmup 10% dos passos, cosseno até 0
               "optim": {"optimizer": "lars", "lr_sqrt": 0.075, "lr_rule": "0,075*sqrt(B)", "warmup": {"frac": 0.10},
                         "start_ratio": 0.0, "end_ratio": 0.0},
               "fiel": ["projetor d->d->128 com BN", "2N vistas numa única chamada", "tau=0,5", "wd 1e-4"],
               "adaptacao": ["aug oficial (crop .08-1, jitter s=1 p=.8, gray .2, blur .5) -> preset 'strong' em LAB",
                             "wd aplicado também a BN/bias (o trainer não separa grupos)",
                             "AdamW no lugar de LARS; lr do growth (não 0,075*sqrt(B))"],
               "pendente": ["LARS", "warmup + cosseno"]},
    "vicreg": {"kwargs": {"weight_decay": 1e-6, "vicreg_kwargs": {"expander_dims": [2048, 2048, 2048],
                                                                  "sim_coeff": 25.0, "std_coeff": 25.0, "cov_coeff": 1.0}},
               # v3.1: LARS, lr = 0,2*B/256 (base 0,2, 1000 ép.), warmup 10 ép. (<= 10% dos passos), cosseno até 1e-3
               "optim": {"optimizer": "lars", "lr_base": 0.2, "lr_rule": "0,2*B/256", "warmup": {"epochs": 10, "max_frac": 0.10},
                         "start_ratio": 0.0, "end_ratio": 1e-3},
               "fiel": ["perda inv/var/cov 25/25/1", "expander 3 camadas Linear-BN-ReLU", "2 vistas"],
               "adaptacao": ["expander 2048 (paper 8192) por memória", "AdamW no lugar de LARS"],
               "pendente": ["LARS", "warmup + cosseno"]},
}


V3_RECIPES.update({
    # v3.1 bloco 7 (articles/barlow_twins.md, byol.md)
    "barlow": {"kwargs": {"weight_decay": 1e-6, "asym_views": True,
                          "barlow_kwargs": {"projector_dims": [2048, 2048, 2048], "lambd": 0.0051,
                                            "lr_w_scale": 0.2, "lr_b_scale": 0.0048}},
               "optim": {"optimizer": "lars", "lr_base": 1.0, "lr_rule": "B/256 (pesos x0,2; bias/BN x0,0048)",
                         "warmup": {"epochs": 10, "max_frac": 0.10}, "start_ratio": 0.0, "end_ratio": 1e-3},
               "fiel": ["perda on + 0,0051*off com BN sem afim por ramo, SOMA em D, fp32", "projetor sem bias",
                        "LARS 2 grupos (0,2 / 0,0048), bias/BN fora do wd e da adaptação", "vistas assimétricas",
                        "wd 1e-6 (código)"],
               "adaptacao": ["projetor 3x2048 (paper 3x8192) = preset dim_reduzido", "solarize/jitter em LAB"],
               "pendente": []},
    "byol": {"kwargs": {"weight_decay": 1.5e-6, "asym_views": True,
                        "byol_kwargs": {"projector_dims": [4096, 256], "predictor_dims": [4096, 256], "tau_base": 0.996}},
             "optim": {"optimizer": "lars", "lr_base": 0.2, "lr_rule": "0,2*B/256", "warmup": {"epochs": 10, "max_frac": 0.10},
                       "start_ratio": 0.0, "end_ratio": 0.0},
             "fiel": ["online = encoder+cabeça+g+q; alvo = EMA de encoder+cabeça+g (sem q), modo train, sem grad",
                      "perda 2-2cos simétrica", "tau_k = 1-(1-0,996)(cos(pi k/K)+1)/2 depois do passo",
                      "LARS, bias/BN fora do wd e da adaptação", "devolve o encoder ONLINE", "vistas assimétricas"],
             "adaptacao": ["solarize/jitter em LAB"], "pendente": []},
    # DINO (articles/dino.md): CNN -> LARS 0,3*B/256, wd 1e-6, min_lr 0,0048/0,3 do pico; professora devolvida
    "dino": {"kwargs": {"weight_decay": 1e-6,
                        "dino_kwargs": {"K": 4096, "hidden": 2048, "bottleneck": 256, "use_bn": True, "n_local": 6,
                                        "local_frac": 96 / 224, "tau_s": 0.1, "tau_t0": 0.04, "tau_t": 0.07,
                                        "tau_t_warmup_epochs": 30, "center_m": 0.9, "momentum_base": 0.996,
                                        "freeze_last_layer_epochs": 1}},
             "optim": {"optimizer": "lars", "lr_base": 0.3, "lr_rule": "0,3*B/256", "warmup": {"epochs": 10, "max_frac": 0.10},
                       "start_ratio": 0.0, "end_ratio": 0.0048 / 0.3},
             "fiel": ["aluna = encoder+cabeça+DINOHead (BN p/ CNN, L2, weight_norm g=1)", "professora EMA (0,996->1),"
                      " só globais, modo train", "centro m=0,9 depois da perda", "tau_s .1; tau_t .04->.07 em 30 ép.",
                      "2 globais assimétricas + 6 locais (96/224 do lado), forward por resolução",
                      "freeze_last_layer 1 ép.", "LARS RN50 (0,3*B/256, wd 1e-6, min_lr 0,0048)",
                      "devolve a PROFESSORA (checkpoint_key=teacher)"],
             "adaptacao": ["K = 4096 (65536 no artigo; App. C)", "jitter/solarize em LAB",
                           "época = passada completa (poucas imagens: warmups em épocas reais)"],
             "pendente": []},
})


def recipe_optim(rec: dict, batch: int, steps_per_epoch: int, planned_epochs: int) -> dict:
    """growth v3.1 (bloco 2): otimizador/lr/agenda da receita para o batch B real e os passos planejados
    (agenda definida sobre a fidelidade MAIS ALTA planejada; a retomada continua pelo passo global)."""
    o = dict(rec.get("optim") or {})
    if not o:
        return {}
    B = int(batch)
    if "lr_sqrt" in o:
        lr = float(o["lr_sqrt"]) * math.sqrt(B)
    elif "lr_base" in o:
        lr = float(o["lr_base"]) * B / 256.0
    else:
        lr = float(o["lr"])
    total = max(1, int(planned_epochs) * int(steps_per_epoch))
    w = o.get("warmup") or {}
    warm = int(w["epochs"]) * int(steps_per_epoch) if "epochs" in w else int(round(float(w.get("frac", 0.0)) * total))
    if "max_frac" in w:
        warm = min(warm, int(round(float(w["max_frac"]) * total)))
    return {"optimizer": o["optimizer"], "lr": lr, "lr_rule": o.get("lr_rule"),
            "lr_schedule": {"total_steps": total, "warmup_steps": int(warm), "end_ratio": float(o.get("end_ratio", 1e-3)),
                            "start_ratio": float(o.get("start_ratio", 0.0))},
            "wd_exclude_1d": bool(o.get("wd_exclude_1d", o["optimizer"] == "adamw"))}


def _ssl_loader(ev, batch: int, seed: int):
    """growth v3.1: loader do SSL com o batch da receita e época = PASSADA COMPLETA do treino (sem o teto de
    imagens por época), shuffle semeado, drop_last (estatísticas de batch sem lote residual)."""
    from torch.utils.data import DataLoader
    ds = ev.train_loader.dataset
    B = min(int(batch), len(ds))
    return DataLoader(ds, batch_size=B, shuffle=True, generator=torch.Generator().manual_seed(int(seed)),
                      drop_last=len(ds) > B, num_workers=0)


def ssl_recipe(gcfg: dict) -> Optional[dict]:
    """Receita do artigo para o SSL efetivo do filho (``v3.ssl_recipe: paper``) ou None (chamada histórica)."""
    v3 = _v3(gcfg)
    if v3.get("ssl_recipe") != "paper" or ssl_disabled(gcfg):
        return None
    r = copy.deepcopy(V3_RECIPES.get(str(gcfg.get("ssl", "lejepa"))))
    if r is None:
        return None
    if r.get("small_views") and v3.get("recipe_small_views"):
        r["kwargs"].update(r["small_views"])
        r["adaptacao"].append(f"V={r['small_views']['n_views']}, lambda={r['small_views']['sigreg_lambda']} por memória")
    variant = str(v3.get("recipe_variant") or "dim_reduzido")   # presets: "fiel_bsN" (expander do artigo) | "dim_reduzido"
    r["variant"] = variant
    if gcfg.get("ssl") == "barlow" and variant.startswith("fiel"):
        r["kwargs"]["barlow_kwargs"]["projector_dims"] = [8192, 8192, 8192]
        r["adaptacao"] = [a for a in r["adaptacao"] if not a.startswith("projetor 3x2048")]
    if gcfg.get("ssl") in ("barlow", "byol", "dino") and v3.get(f"{gcfg.get('ssl')}_kwargs"):
        k_ = f"{gcfg.get('ssl')}_kwargs"
        r["kwargs"][k_] = {**r["kwargs"][k_], **v3[k_]}
    if gcfg.get("ssl") == "vicreg" and variant.startswith("fiel"):
        r["kwargs"]["vicreg_kwargs"]["expander_dims"] = [8192, 8192, 8192]
        r["adaptacao"] = [a for a in r["adaptacao"] if not a.startswith("expander 2048")]
    if gcfg.get("ssl") == "lejepa" and variant.startswith("fiel"):
        pass                                                 # projetor 2048-2048-128 já é o do artigo
    if gcfg.get("ssl") == "vicreg" and v3.get("vicreg_kwargs"):
        r["kwargs"]["vicreg_kwargs"] = {**r["kwargs"]["vicreg_kwargs"], **v3["vicreg_kwargs"]}
    return r


def _v3(gcfg: dict) -> dict:
    base = copy.deepcopy(GROWTH_DEFAULTS["v3"])
    for k, v in (gcfg.get("v3") or {}).items():
        base[k] = {**base[k], **v} if isinstance(base.get(k), dict) and isinstance(v, dict) else v
    return base


def so2_on(gcfg: dict) -> bool:
    return str(_v3(gcfg).get("readout", "head")) != "head"


def gene_space(gcfg: dict) -> List[dict]:
    """Produto cartesiano dos genes v3 com lista (ordem fixa); gene ausente/escalar = fixo."""
    import itertools
    genes = _v3(gcfg).get("genes") or {}
    keys = [k for k in ("ssl", "aug", "whiten_new", "softhebb", "spifil_images_per_class", "spifil_n_superpixels")
            if isinstance(genes.get(k), list) and genes[k]]
    if not keys:
        return [{}]
    return [dict(zip(keys, combo)) for combo in itertools.product(*[genes[k] for k in keys])]


def child_genes(gcfg: dict, tag: str, i: int, seed: int = 0) -> dict:
    """Genes do i-ésimo filho da geração ``tag``: rodízio determinístico no produto cartesiano com deslocamento
    por geração (cobre todas as combinações ao longo da cadeia). {} quando não há genes."""
    space = gene_space(gcfg)
    if space == [{}]:
        return {}
    off = _h({"genes_tag": tag, "seed": int(seed)}) % len(space)
    return dict(space[(off + int(i)) % len(space)])


def apply_genes(gcfg: dict, genes: dict) -> dict:
    """gcfg efetivo de um filho: ssl/reg, aug, whiten_new, softhebb."""
    if not genes:
        return gcfg
    g = copy.deepcopy(gcfg)
    if "ssl" in genes:
        s = genes["ssl"]
        g["ssl"] = "none" if s in (None, False, "none", "ae", "ae_local_only") else str(s)
        if g["ssl"] in V3_SSL_REG:
            g["reg"] = V3_SSL_REG[g["ssl"]]
    if "spifil_images_per_class" in genes:   # "all" -> teto do pool estratificado de treino (spifil_k_max)
        v = genes["spifil_images_per_class"]
        g["spifil_images_per_class"] = 10 ** 6 if str(v).lower() in ("all", "todas") else int(v)
    g.setdefault("v3", {})
    for k in ("aug", "whiten_new", "softhebb", "spifil_n_superpixels"):
        if k in genes:
            g["v3"] = {**g["v3"], k: genes[k]}
    return g


def ae_extra_kw(gcfg: dict) -> dict:
    """growth_v31_hotfix_ae_diverge: kwargs do ``train_local_ae`` a partir do growth.yaml. Valores ausentes/
    desligados não entram (chamada idêntica à histórica)."""
    kw = {}
    if gcfg.get("ae_dec_in_norm"):
        kw["dec_in_norm"] = True
    if gcfg.get("ae_grad_clip") not in (None, 0, 0.0, False):
        kw["grad_clip"] = float(gcfg["ae_grad_clip"])
    if gcfg.get("ae_lr_scale_fanin"):
        kw["lr_scale_fanin"] = True
    if int(gcfg.get("ae_warmup_steps") or 0) > 0:
        kw["warmup_steps"] = int(gcfg["ae_warmup_steps"])
    if gcfg.get("ae_diverge_guard") not in (None, False):
        kw["diverge_ratio"] = float(gcfg["ae_diverge_guard"])
    return kw


def ae_smoke_strict(gcfg: dict) -> bool:
    v = gcfg.get("ae_smoke_strict")
    if v is None:
        return int(gcfg.get("fid_high", 0) or 0) <= 1
    return bool(v)


def _mem(gcfg: dict) -> dict:
    return {**GROWTH_DEFAULTS["mem"], **(gcfg.get("mem") or {})}


def _promo(gcfg: dict) -> dict:
    p = {**GROWTH_DEFAULTS["promotion"], **(gcfg.get("promotion") or {})}
    if p["mode"] not in ("resume", "retrain"):
        raise ValueError(f"promotion.mode {p['mode']!r} (resume|retrain)")
    if p["extra_epochs"] not in ("fid_diff", "ssl_diff"):
        raise ValueError(f"promotion.extra_epochs {p['extra_epochs']!r} (fid_diff|ssl_diff)")
    return p


def _free(dev, on: bool = True) -> None:
    import gc
    gc.collect()
    if on and getattr(dev, "type", str(dev)) == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


_STAT_MODS = (nn.modules.batchnorm._BatchNorm,)


class _FrozenStats:
    """Na RECOMPUTAÇÃO do checkpoint: momentum 0 nos BN/DBN (running = 1*running + 0*batch, bit a bit) e
    num_batches_tracked restaurado -> as estatísticas correntes são atualizadas uma única vez, como sem checkpoint."""

    def __init__(self, mod: nn.Module):
        from ga_search.modehb.ops_norm_act import DecorrelatedBatchNorm
        self.mods = [m for m in mod.modules() if isinstance(m, _STAT_MODS + (DecorrelatedBatchNorm,))]

    def __enter__(self):
        self.saved = [(m.momentum, m.num_batches_tracked.clone() if getattr(m, "num_batches_tracked", None)
                       is not None else None) for m in self.mods]
        self.bufs = [{k: v.clone() for k, v in m.named_buffers(recurse=False)} for m in self.mods]
        for m in self.mods:
            m.momentum = 0.0

    def __exit__(self, *exc):
        with torch.no_grad():
            for m, (mom, _), bufs in zip(self.mods, self.saved, self.bufs):
                m.momentum = mom
                for k, v in m.named_buffers(recurse=False):   # garantia bit a bit (0*inf etc.)
                    v.copy_(bufs[k])
        return False


class _ActPool(torch.autograd.Function):
    """``pool(act(y) [+ s])`` em micro-lotes de amostras, salvando SÓ ``y`` (e ``s``).

    Act sem parâmetros (elementwise) e pool são por amostra: o micro-lote no eixo N dá exatamente os
    mesmos números (mesmos kernels, mesma ordem por elemento). Evita guardar a saída da ativação e os
    índices int64 do MaxPool em resolução cheia (no bloco 0 de RS: 2 x 9 GiB a 256 imagens 224 px)."""

    @staticmethod
    def forward(ctx, y, s, act, pool, chunk):
        ctx.act, ctx.pool, ctx.chunk, ctx.has_s = act, pool, int(chunk), s is not None
        outs = []
        for i in range(0, y.shape[0], ctx.chunk):
            z = act(y[i:i + ctx.chunk])
            if s is not None:
                z = z + s[i:i + ctx.chunk]
            outs.append(pool(z))
        ctx.save_for_backward(y, s) if s is not None else ctx.save_for_backward(y)
        return torch.cat(outs, 0)

    @staticmethod
    def backward(ctx, g):
        y, s = (ctx.saved_tensors + (None,))[:2]
        gy = gs = None          # layout do grad = o da 1ª fatia (o que o autograd produziria): a redução
                                # a jusante (ex.: grad do afim do DBN) depende dos strides -> bit a bit
        for i in range(0, y.shape[0], ctx.chunk):
            with torch.enable_grad():
                yc = y[i:i + ctx.chunk].detach().requires_grad_(True)
                ins = [yc]
                z = ctx.act(yc)
                if s is not None:
                    sc = s[i:i + ctx.chunk].detach().requires_grad_(True)
                    ins.append(sc)
                    z = z + sc
                o = ctx.pool(z)
                grads = torch.autograd.grad(o, ins, g[i:i + ctx.chunk])
            if gy is None:
                fmt = lambda t, ref: torch.empty(ref.shape, dtype=t.dtype, device=t.device) \
                    if t.is_contiguous() else torch.empty_like(ref)
                gy = fmt(grads[0], y)
                gs = fmt(grads[1], s) if s is not None else None
            gy[i:i + ctx.chunk] = grads[0]
            if s is not None:
                gs[i:i + ctx.chunk] = grads[1]
        return gy, gs, None, None, None


ACTPOOL_CHUNK = 16


def _fusable(blk) -> bool:
    from ga_search.modehb.spifil_ops import SpifilOpsBlock
    return (type(blk) is SpifilOpsBlock and isinstance(blk.dropout, nn.Identity)
            and not any(True for _ in blk.activation.parameters()) and not any(True for _ in blk.pool.parameters())
            and (blk.skip is None or isinstance(blk.skip, nn.Identity)))


def _block_forward(blk, x):
    """== ``SpifilOpsBlock.forward`` (mesmas ops), com act+skip+pool por ``_ActPool``."""
    if not _fusable(blk):
        return blk(x)
    y = blk.norm(blk.conv(x))
    s = x if blk.skip is not None else None
    return _ActPool.apply(y, s, blk.activation, blk.pool, ACTPOOL_CHUNK)


class _CkptSequential(nn.Sequential):
    """``nn.Sequential`` com os MESMOS filhos e nomes ('0.conv.weight'...), checkpoint por bloco no treino."""

    def forward(self, x):
        from torch.utils.checkpoint import checkpoint
        if not (self.training and torch.is_grad_enabled()):
            return super().forward(x)
        for blk in self:
            calls = [0]

            def run(inp, blk=blk, calls=calls):
                calls[0] += 1
                if calls[0] == 1:
                    return _block_forward(blk, inp)
                with _FrozenStats(blk):
                    return _block_forward(blk, inp)
            x = checkpoint(run, x, use_reentrant=False, preserve_rng_state=True)
        return x


# ═════════════════════════════════════════════════════════════════ growth_fix5: estimador de pico de memória
# Pico do SSL global (checkpoint por bloco + _ActPool do mem3) e do AE local, em GB (1e9), ANTES de treinar.
# Unidade: elementos N·C·H·W de cada bloco na SUA resolução de entrada (pré-pool), N = V·B no SSL e B no AE.
#   saved   = entradas dos blocos guardadas pelo checkpoint (bytes do autocast);
#   recomp  = o MAIOR bloco recomputado no backward: K_CONV cópias (saída da conv, y salvo pelo _ActPool, grad
#             de y) nos bytes do autocast + cópias da norma: BN (cuDNN, bytes do autocast) ou DBN/ZCA (que sobem
#             para fp32 sob bf16: x.float(), flat permutado, xc, grupos branqueados/cat) + dwsep: depthwise
#             intermediário com Cin canais; bloco não fundível (sem _ActPool): + saída da ativação + índices
#             int64 do MaxPool;
#   head    = projeção 1x1 da cabeça na resolução final (saída + grad + momentos);
#   c0      = contexto CUDA + pesos/otimizador + SIGReg + fragmentação do alocador.
# Calibrado (tests/hmodehb/test_growth_fix5.py) contra os picos medidos no athena/jaci (±20%).
MEM_EST = {"c0_gb": 2.0, "k_conv": 3.0, "k_norm": {"batch": 1.0, "batch_dbn": 5.0, "dbn": 5.0, "zca": 1.0, "batch_zca": 2.0},
           "k_norm_default": 1.0, "fp32_norms": ("batch_dbn", "dbn"), "k_dw": 2.0, "k_unfused": 2.0,
           "k_head": 3.0, "unfused_idx_bytes": 8}


def _bytes_of(amp_bf16: bool) -> int:
    return 2 if amp_bf16 else 4


def _head_proj_channels(model) -> int:
    for m in model.head.modules():
        if isinstance(m, nn.Conv2d):
            return int(m.out_channels)
    return int(getattr(model.spec.head, "embed_dim", 1280))


def _block_geometry(model) -> List[dict]:
    """Por bloco: Cin, Cout, resolução de entrada H (pool stride 2 depois), conv/norm/pool/fusível."""
    spec = model.spec
    H = int(spec.input.resolution)
    cin = int(spec.input.in_channels)
    out = []
    blocks = list(getattr(model, "blocks", []))
    for j, L in enumerate(spec.layers()):
        fus = _fusable(blocks[j]) if j < len(blocks) else True
        out.append({"cin": cin, "cout": int(L.channels), "H": H, "conv": str(L.conv), "norm": str(L.norm),
                    "pool": str(L.pool), "fusable": bool(fus)})
        if L.pool != "none":
            H = (H + 1) // 2
        cin = _block_out(spec, j)
    return out


def _block_recompute_bytes(g: dict, n: int, b: int) -> float:
    E = MEM_EST
    el = float(n) * g["cout"] * g["H"] * g["H"]
    norm_b = 4 if (g["norm"] in E["fp32_norms"] and b < 4) else b
    k_norm = E["k_norm"].get(g["norm"], E["k_norm_default"]) if g["norm"] != "none" else 0.0
    tot = el * (E["k_conv"] * b + k_norm * norm_b)
    if g["conv"] in ("dwsep", "depthwise"):
        tot += float(n) * g["cin"] * g["H"] * g["H"] * E["k_dw"] * b
    if not g["fusable"]:
        tot += el * E["k_unfused"] * b + (el * E["unfused_idx_bytes"] if g["pool"] == "max" else 0.0)
    return tot


def estimate_peak_gb(model, *, batch: int, views: int = 4, amp_bf16: bool = True, ae_amp_bf16: Optional[bool] = None,
                     new_blocks: Optional[Sequence[int]] = None, ssl: bool = True,
                     grad_checkpoint: bool = True) -> dict:
    """Pico estimado (GB) do SSL global e do AE local de um plano (modelo-filho montado na CPU), antes de treinar.

    ``new_blocks``: blocos que passam pelo AE local (None = todos). Devolve {"ssl_gb", "ae_gb", "peak_gb",
    "argmax_block", ...}. Monótono em B, V, canais e resolução."""
    E = MEM_EST
    geo = _block_geometry(model)
    b = _bytes_of(amp_bf16)
    b_ae = _bytes_of(amp_bf16 if ae_amp_bf16 is None else ae_amp_bf16)
    N = int(views) * int(batch)
    ssl_gb, arg = 0.0, None
    if ssl and geo:
        saved = sum(float(N) * g["cin"] * g["H"] ** 2 * b for g in geo)
        rec = [_block_recompute_bytes(g, N, b) for g in geo]
        if grad_checkpoint:
            top = max(rec)
        else:                               # sem checkpoint: tudo fica salvo
            top = sum(rec)
        arg = int(np.argmax(rec))
        last = geo[-1]
        Hl = (last["H"] + 1) // 2 if last["pool"] != "none" else last["H"]
        head = float(N) * _head_proj_channels(model) * Hl * Hl * b * E["k_head"]
        ssl_gb = (saved + top + head) / 1e9 + E["c0_gb"]
    nb = list(range(len(geo))) if new_blocks is None else [int(j) for j in new_blocks]
    ae_gb = 0.0
    for j in nb:
        if 0 <= j < len(geo):
            g = geo[j]
            dec = float(batch) * int(model.spec.input.in_channels) * int(model.spec.input.resolution) ** 2 * b_ae * 2
            v = (_block_recompute_bytes(g, int(batch), b_ae) + float(batch) * g["cin"] * g["H"] ** 2 * b_ae + dec) / 1e9
            ae_gb = max(ae_gb, v + E["c0_gb"])
    return {"ssl_gb": float(ssl_gb), "ae_gb": float(ae_gb), "peak_gb": float(max(ssl_gb, ae_gb)),
            "argmax_block": arg, "batch": int(batch), "views": int(views), "bytes": b, "bytes_ae": b_ae}


def resolve_max_gpu_gb(gcfg: dict, device=None) -> Optional[float]:
    """``growth.mem.max_gpu_gb``; padrão (null) = memória total da GPU - 3 GB (torch.cuda) ou None (sem GPU:
    filtro desligado)."""
    v = _mem(gcfg).get("max_gpu_gb")
    if v is not None:
        return float(v)
    try:
        if torch.cuda.is_available() and (device is None or getattr(device, "type", str(device)) == "cuda"):
            idx = getattr(device, "index", None)
            idx = torch.cuda.current_device() if idx is None else idx
            return torch.cuda.get_device_properties(idx).total_memory / 1e9 - 3.0
    except Exception:
        pass
    return None


def mem_ctx_from(cfg: dict, gcfg: dict, device=None) -> dict:
    """Contexto do estimador: batch do treino, V do SSL, bytes (bf16 do treino ou ``mem.ssl_amp_bf16``)."""
    tr = (cfg or {}).get("train") or {}
    amp = str(tr.get("amp_dtype") or "").lower() in ("bf16", "bfloat16")
    views = {"simclr": 2, "lejepa": 4, "ae": 1, "vicreg": 2, "barlow": 2, "byol": 2, "dino": 8}.get(str(gcfg.get("ssl", "lejepa")).lower(), 4)
    rec = ssl_recipe(gcfg)                    # growth v3: a receita do artigo pode mudar V (LeJEPA V=8)
    if rec is not None and rec["kwargs"].get("n_views"):
        views = int(rec["kwargs"]["n_views"])
    return {"batch": int(((cfg or {}).get("data") or {}).get("batch_size", 64)), "views": int(views),
            "amp_bf16": bool(amp or _mem(gcfg).get("ssl_amp_bf16")), "ae_amp_bf16": bool(amp),
            "ssl": not ssl_disabled(gcfg), "grad_checkpoint": bool(_mem(gcfg).get("grad_checkpoint", True)),
            "max_gb": resolve_max_gpu_gb(gcfg, device)}


def plan_memory(parent, ops: Sequence[dict], ctx: dict) -> dict:
    """Pico estimado do filho de ``parent`` pelo plano ``ops`` (montado na CPU com init PyTorch: a forma é a
    mesma do realize_plan com SPiFiL)."""
    res = realize_plan(parent, ops, None)
    model = res.child
    masks = res.masks
    new_blocks = [j for j in range(len(model.blocks))
                  if any(not bool(masks[f"blocks.{j}.{k}"].all()) for k, _ in model.blocks[j].named_parameters()
                         if f"blocks.{j}.{k}" in masks)]
    est = estimate_peak_gb(model, batch=ctx["batch"], views=ctx["views"], amp_bf16=ctx["amp_bf16"],
                           ae_amp_bf16=ctx.get("ae_amp_bf16"), new_blocks=new_blocks, ssl=ctx.get("ssl", True),
                           grad_checkpoint=ctx.get("grad_checkpoint", True))
    est["max_gb"] = ctx.get("max_gb")
    est["feasible"] = ctx.get("max_gb") is None or est["peak_gb"] <= float(ctx["max_gb"])
    return est


# growth_fix5 (3): OOM de CAPACIDADE (o filho não cabe na placa nem vazia) -> failed na hora, sem retries.
_GIB = 1024 ** 3


def _parse_gib(msg: str, pat: str) -> Optional[float]:
    import re
    m = re.search(pat, msg)
    if not m:
        return None
    val, unit = float(m.group(1)), m.group(2)
    return val * {"GiB": 1.0, "MiB": 1 / 1024, "KiB": 1 / 1024 ** 2, "bytes": 1 / _GIB}.get(unit, 1.0)


def oom_info(exc_or_msg) -> dict:
    """Campos da mensagem de OOM do PyTorch (GiB): pedido, total, livre, uso deste processo."""
    msg = str(exc_or_msg)
    u = r"([\d.]+) (GiB|MiB|KiB|bytes)"
    return {"requested": _parse_gib(msg, r"Tried to allocate " + u),
            "total": _parse_gib(msg, r"total capacity of " + u),
            "free": _parse_gib(msg, r"of which " + u + r" is free"),
            "this_process": _parse_gib(msg, r"this process has " + u)}


def oom_is_capacity(exc_or_msg, *, total_gib: Optional[float] = None, margin_gib: float = 1.0,
                    alone_tol_gib: float = 0.75) -> Tuple[bool, dict]:
    """True quando o pedido não cabe nem com a GPU só para este processo:
    (a) pedido + uso deste processo > total - margem; ou (b) a GPU não tem outros processos (total - livre -
    este <= tolerância) e o pedido > livre. Sem os números na mensagem: False (trata como contenção)."""
    d = oom_info(exc_or_msg)
    tot = d["total"] if d["total"] is not None else total_gib
    req, this, free = d["requested"], d["this_process"], d["free"]
    if req is None or tot is None:
        return False, d
    cap = this is not None and req + this > tot - float(margin_gib)
    if not cap and free is not None and this is not None:
        others = tot - free - this
        cap = others <= float(alone_tol_gib) and req > free
    d["others"] = (tot - free - this) if (free is not None and this is not None) else None
    return bool(cap), d


class ChainError(RuntimeError):
    pass


class SpifilFallback(ChainError):
    """growth v3 (regra do usuário: init SEMPRE SPiFiL): algum filtro novo cairia em init PyTorch -> plano infeasible."""


class PreservationViolated(ChainError):
    """O morfismo declarado exato não preservou a função (bug): o filho falha, nunca é treinado."""


# ═════════════════════════════════════════════════════════════════ 1. checkpoints de produção -> Backbone
def _layer_from_effective(eff: dict) -> LayerSpec:
    if eff.get("builder") not in ("spifil_ops", None) or not eff.get("norm_in_block", True):
        raise ChainError(f"bloco {eff.get('builder')!r} não suportado (só spifil_ops com norm no bloco)")
    if float(eff.get("dropout", 0.0) or 0.0) > 0:
        raise ChainError("dropout > 0 não suportado no crescimento")
    skip = eff.get("skip")
    residual = {"identity": "identity", "project": "project", None: "none"}[skip]
    L = LayerSpec(channels=int(eff["channels"]), kernel=int(eff["kernel"]), conv=str(eff["conv"]),
                  norm=str(eff.get("norm", "none")), activation=str(eff.get("activation", "relu")),
                  residual=residual, pool=str(eff.get("pool", "none")), init="inherited", flavor="spifil_ops")
    if eff.get("conv") == "inception":
        L.branches = [[str(n), int(k), int(c)] for n, k, c in eff["inception_branches"]]
    if eff.get("pool_padding") is not None and L.pool != "none":
        L.pool_padding, L.pool_size = int(eff["pool_padding"]), int(eff.get("pool_size", 3))
    if L.pool != "none" and int(eff.get("pool_stride", 2)) != 2:
        raise ChainError("pool_stride != 2 não suportado")
    return L


def spec_from_layers(layers: List[LayerSpec], head: HeadSpec, dataset: str, image_size: int) -> BackboneSpec:
    stages = [StageSpec("local", list(layers[1:]))] if len(layers) > 1 else []
    return BackboneSpec(stem=layers[0], stages=stages, head=head, input=InputSpec(resolution=int(image_size)),
                        dataset=dataset)


def _head_spec(head: nn.Module) -> HeadSpec:
    from ga_search.modehb.pooling_head import EmbeddingHead
    if not isinstance(head, EmbeddingHead):
        raise ChainError(f"cabeça {type(head).__name__} não suportada")
    bn = any(isinstance(m, nn.BatchNorm2d) for m in head.proj.modules())
    acts = [m for m in head.proj.modules() if not isinstance(m, (nn.Sequential, nn.Conv2d, nn.BatchNorm2d))]
    if acts:
        raise ChainError("cabeça com ativação não suportada")
    kurt = getattr(head, "n_blocks", 4) == 4 if head.pooling == "moments" else True
    return HeadSpec(embed_dim=int(head.nominal_dim), pooling=str(head.pooling), kurtosis=bool(kurt), bn=bn)


@torch.no_grad()
def backbone_from_production(path, dataset: str, image_size: int, verify_x: Optional[torch.Tensor] = None):
    """Encoder ``nn.Sequential(SpifilOpsBlock...)`` + ``EmbeddingHead`` da busca -> ``Backbone`` A3 (bit a bit)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    enc, head, effs = ck["encoder"], ck["head"], list(ck.get("effective_specs") or [])
    if len(effs) != len(enc):
        raise ChainError(f"{path}: effective_specs ({len(effs)}) != blocos ({len(enc)})")
    layers = [_layer_from_effective(e) for e in effs]
    spec = spec_from_layers(layers, _head_spec(head), dataset, image_size)
    model = BB.Backbone(spec)
    for j, blk in enumerate(enc):
        model.blocks[j].load_state_dict(blk.state_dict(), strict=True)
    model.head.load_state_dict(head.state_dict(), strict=True)
    model.spec.cost = BB.measure_cost(model)
    model.spec.budget_class = band_of(model.spec.cost.params_total)
    info = {"source": str(path), "genome": ck.get("genome"), "fidelity": ck.get("fidelity"),
            "cand_id": ck.get("cand_id"), "conversion_max_abs": None}
    if verify_x is not None:
        enc.eval(), head.eval(), model.eval()
        a, b = head(enc(verify_x)), model(verify_x)
        d = float((a - b).abs().max())
        info["conversion_max_abs"], info["conversion_bitwise"] = d, bool(torch.equal(a, b))
        if d > 1e-6 * max(1.0, float(a.abs().max())):
            raise ChainError(f"{path}: conversão não preserva o embedding (max|d|={d:.3g})")
    return model.eval(), info


@torch.no_grad()
def backbone_from_external_small(dataset: str, image_size: int, *, split: int = 1, seed: int = 42,
                                 head: Optional[HeadSpec] = None):
    """Semente "SPiFiL pequeno" (A2 ``seed_small``; FLIM L0 + LeJEPA de ``artifacts/external_ssl_pequeno``)
    com os PESOS importados (conv/bias) e cabeça >= 1280 nova (semente registrada)."""
    from ga_search.modehb import external_seed as ES
    genome, blocks = ES.load_external_seed(dataset, None, split=split, seed=seed)
    layers = [_layer_from_effective(b["spec"]) for b in blocks]
    hs = head or HeadSpec(embed_dim=int(genome.get("embed_dim", 1280)), pooling=str(genome.get("pooling", "gem")))
    spec = spec_from_layers(layers, hs, dataset, image_size)
    model = BB.build_backbone(spec, seed=seed, enforce_budget=False)
    for j, b in enumerate(blocks):
        model.blocks[j].load_state_dict(b["state_dict"], strict=True)
    return model.eval(), {"source": f"external_small:{genome.get('external', {}).get('name')}",
                          "genome": genome, "head": "nova (init PyTorch, seed registrada)", "seed": seed}


# ═════════════════════════════════════════════════════════════════ 2. avaliador (dados + régua ORIG)
@contextlib.contextmanager
def fp32_reading(ev):
    """growth_fix4 (a): leitura determinística para a SVM. Autocast desligado (``ev.embed_amp=False``) e TF32
    desligado em conv/matmul enquanto o embedding é extraído; estado anterior restaurado na saída."""
    old_amp = getattr(ev, "embed_amp", True)
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    mm_tf32 = torch.backends.cuda.matmul.allow_tf32
    ev.embed_amp = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        ev.embed_amp = old_amp
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = mm_tf32


class GrowthEvaluator:
    """Dados e régua ORIG do ``HEvaluator`` (A4): ``head(encoder(x))`` >= 1280, core.svm.fit_svm C=100 sem scaler."""

    def __init__(self, cfg: dict, out_dir, debug_train: Optional[int] = None, debug_val: Optional[int] = None,
                 v3: Optional[dict] = None):
        from ga_search.hmodehb.hevaluate import HEvaluator
        from ga_search.modehb.run_search import _stratified_indices
        from torch.utils.data import DataLoader, TensorDataset
        self.cfg = cfg
        self.ev = ev = HEvaluator(cfg, out_dir)
        if debug_train:                          # SÓ TESTE (como o A2): subconjunto estratificado do treino
            idx = torch.as_tensor(_stratified_indices(ev.svm_train_y.numpy(), int(debug_train), ev.seed))
            ev.svm_train_x, ev.svm_train_y = ev.svm_train_x[idx], ev.svm_train_y[idx]
            bs = int(cfg["data"].get("batch_size", 64))
            ds = TensorDataset(ev.svm_train_x)
            ev.train_loader = DataLoader(ds, batch_size=bs, shuffle=True,
                                         generator=torch.Generator().manual_seed(ev.seed), drop_last=len(ds) > bs)
        if debug_val:
            idx = torch.as_tensor(_stratified_indices(ev.val_y.numpy(), int(debug_val), ev.seed))
            ev.val_x, ev.val_y = ev.val_x[idx], ev.val_y[idx]
        self.device = ev.device
        ev.embed_amp = False                     # growth_fix4 (a): run_search._embed sem autocast (fp32)
        self.embed_precision = EMBED_PRECISION
        self.fitness_spec = FV.spec(ev.sv2, val_sha256=ev.split_hash, n_val=int(len(ev.val_y)),
                                    dataset=ev.dataset, split=(cfg.get("data") or {}).get("split"))
        self.fitness_spec["embed_precision"] = EMBED_PRECISION
        # growth v3 (1): readout de 2ª ordem fixa na fitness (e no teste). Entra na fitness_spec -> fitness_sig nova.
        v3 = _v3({"v3": v3 or {}})
        self.readout = str(v3.get("readout", "head"))
        self.so2_cfg = dict(v3.get("so2") or {})
        self.aux_gap_on = bool(v3.get("aux_gap", True))
        if v3.get("accept_bootstrap"):          # v3.1: a SVM devolve as predições da val (bootstrap pareado)
            ev.return_pred = True
        self._last_head = None
        if self.readout != "head":
            from ga_search.hmodehb import so2_head as SO2
            if self.readout != SO2.READOUT and not self.readout.startswith("so2_"):
                raise FV.FitnessConfigError(f"v3.readout {self.readout!r} (head|so2_4096_pca)")
            self.fitness_spec["readout"] = self.readout
            self.fitness_spec["head_v3"] = SO2.spec_entry(int(self.so2_cfg.get("dim", 4096)),
                                                          int(self.so2_cfg.get("n_px", 64)))
            if float(self.so2_cfg.get("eps_rel", 0.0) or 0.0) > 0:    # v3.1: Cov + eps*(tr/k)*I (muda a régua)
                self.fitness_spec["head_v3"]["eps_rel"] = float(self.so2_cfg["eps_rel"])
            if str(self.so2_cfg.get("seed_mode", "block_index")) != "block_index":
                # v3.1: semente da amostragem de pixels da PCA FIXA (não depende da profundidade): um bloco
                # inserido com g=0 dá exatamente a mesma leitura (sem ruído de leitura disfarçado de ganho)
                self.fitness_spec["head_v3"]["seed_mode"] = str(self.so2_cfg["seed_mode"])
        self.fitness_sig = FV.signature(self.fitness_spec)
        self.reset_stamps()

    def reset_stamps(self):
        self.ev._readouts, self.ev._svm_stamps = [], []

    def decision_rulers(self) -> List[str]:
        return sorted({FV.ruler_sig({"readout": r, "tta": 1, "solver": "libsvm", **st})
                       for r in set(self.ev._readouts) for st in self.ev._svm_stamps})

    def so2_read(self, model: BB.Backbone, others: Sequence[torch.Tensor] = (), tta: int = 1, gap: bool = False,
                 embed_batch_size: Optional[int] = None, head=None):
        """growth v3 (1): cabeça de 2ª ordem fixa (PCA SÓ no treino) -> (So2Head, [z_train, z_others...], gaps)."""
        from ga_search.hmodehb import so2_head as SO2
        model.to(self.device)
        enc = nn.Sequential(*model.blocks)
        bs = int(embed_batch_size or self.ev.cfg["data"].get("batch_size", 64))
        with fp32_reading(self.ev), torch.no_grad():
            head, outs, gaps = SO2.fit_and_read(enc, self.ev.svm_train_x, list(others), self.device, batch_size=bs,
                                                dim=int(self.so2_cfg.get("dim", 4096)),
                                                n_px=int(self.so2_cfg.get("n_px", 64)),
                                                block_index=(len(model.blocks) - 1 if str(self.so2_cfg.get(
                                                    "seed_mode", "block_index")) == "block_index" else 0),
                                                tta=int(tta),
                                                gap_head=model.head if gap else None, head=head,
                                                eps_rel=float(self.so2_cfg.get("eps_rel", 0.0) or 0.0))
        _free(self.device)
        return head, outs, gaps

    def aux_gap(self, model: BB.Backbone) -> Optional[dict]:
        """growth v3: kappa ORIG-val da cabeça antiga (GAP/proj) como coluna AUX; os carimbos de decisão não mudam."""
        if getattr(self, "readout", "head") == "head" or not getattr(self, "aux_gap_on", False):
            return None
        from ga_search.modehb.run_search import _blas_threads
        saved = (list(self.ev._readouts), [dict(x) for x in self.ev._svm_stamps])
        try:
            model.to(self.device)
            with fp32_reading(self.ev):
                feats = self.ev._svm_features(nn.Sequential(*model.blocks), model.head)
            with _blas_threads(self.ev._fitness_speed()["svm_blas_threads"]):
                sv = self.ev._svm_score(*feats)
            return {"kappa_val_gap": float(sv["kappa"]), "macro_f1_val_gap": float(sv.get("macro_f1", float("nan"))),
                    "readout": "head(gap/proj)", "used_for_selection": False}
        finally:
            self.ev._readouts, self.ev._svm_stamps = saved
            _free(self.device)

    def features(self, model: BB.Backbone, embed_batch_size: Optional[int] = None):
        """growth_fix7: parte GPU do ``score`` (leitura fp32 determinística) -> (train, val)."""
        if getattr(self, "readout", "head") != "head":   # growth v3 (1): so2 (a cabeça ajustada fica em self._last_head)
            head, (tr, va), _ = self.so2_read(model, [self.ev.val_x], embed_batch_size=embed_batch_size)
            self.ev.__dict__.setdefault("_readouts", []).append(self.readout)
            self._last_head = head
            return tr.numpy(), va.numpy()
        model.to(self.device)
        enc = nn.Sequential(*model.blocks)
        data = self.ev.cfg["data"]
        old_bs = data.get("batch_size", None)
        if embed_batch_size:                  # OPÇÃO DESLIGADA (null): micro-lote só na extração da SVM
            data["batch_size"] = int(embed_batch_size)
        try:
            with fp32_reading(self.ev):
                feats = self.ev._svm_features(enc, model.head)
        finally:
            if embed_batch_size:
                if old_bs is None:
                    data.pop("batch_size", None)
                else:
                    data["batch_size"] = old_bs
        _free(self.device)
        return feats

    def svm(self, feats) -> dict:
        """growth_fix7: parte CPU do ``score`` (a SVM do repo, inalterada: libsvm linear, C=100, sem scaler).
        Sem o contexto de threads BLAS: quem chama abre ``_blas_threads`` (sequencial ou em lote)."""
        sv = self.ev._svm_score(*feats)
        if sv.get("nonfinite"):
            raise FV.NonFiniteFitness("embedding não finito")
        sv["embed_precision"] = self.embed_precision
        return sv

    def score(self, model: BB.Backbone, embed_batch_size: Optional[int] = None) -> dict:
        from ga_search.modehb.run_search import _blas_threads
        feats = self.features(model, embed_batch_size)
        with _blas_threads(self.ev._fitness_speed()["svm_blas_threads"]):
            return self.svm(feats)

    def score_many(self, feats_list: Sequence) -> List[dict]:
        """growth_fix7: as SVMs de várias leituras em PARALELO (threads; o fit do libsvm roda sem o GIL).
        Cada fit é a mesma conta determinística do ``score`` sequencial (mesmos dados, mesma SVM, mesma
        thread BLAS), então o resultado é idêntico; a ordem da saída é a da entrada.
        Workers = ``eval.svm_fitness.svm_workers`` (padrão min(#leituras, threads do torch)), limitado pela RAM
        (``svm_ram_gb``, como no ``run_search``)."""
        from concurrent.futures import ThreadPoolExecutor
        from ga_search.modehb.run_search import _blas_threads
        feats_list = list(feats_list)
        if not feats_list:
            return []
        sp = self.ev._fitness_speed(len(feats_list))
        workers = self.ev._svm_workers_for(sp["svm_workers"], np.asarray(feats_list[0][0]))
        with _blas_threads(sp["svm_blas_threads"]):
            if workers <= 1 or len(feats_list) == 1:
                return [self.svm(f) for f in feats_list]
            with ThreadPoolExecutor(min(workers, len(feats_list)), thread_name_prefix="growth_svm") as ex:
                return list(ex.map(self.svm, feats_list))

    def score_parent(self, model: BB.Backbone, key: str) -> Tuple[dict, List[str]]:
        """growth_fix7: re-score do pai com cache no processo por (sha256 do checkpoint, leitura, régua, split).
        A leitura é determinística (fix4 a), então a 2ª vez daria o mesmo κ; aqui ela não é refeita.
        Devolve (sv, decision_rulers) e deixa os carimbos como se o score tivesse rodado."""
        ck = (str(key), self.embed_precision, self.fitness_sig, len(self.ev.val_y), len(self.ev.svm_train_y))
        cache = self.__dict__.setdefault("_parent_cache", {})
        self.reset_stamps()
        if ck in cache:
            sv, stamps, readouts = cache[ck]
            self.ev._svm_stamps = [dict(x) for x in stamps]
            self.ev._readouts = list(readouts)
            return dict(sv, parent_rescore_cached=True), self.decision_rulers()
        sv = self.score(model)
        cache[ck] = (dict(sv), [dict(x) for x in self.ev._svm_stamps], list(self.ev._readouts))
        return dict(sv, parent_rescore_cached=False), self.decision_rulers()

    def spifil_cfg(self, seed: int) -> dict:
        c = {**(self.cfg.get("spifil") or {}), "device": str(self.device), "seed": int(seed),
             "image_size": int(self.cfg["data"].get("image_size", 224))}
        c.pop("in_channels", None)
        return c


# ═════════════════════════════════════════════════════════════════ 3. SPiFiL nos canais novos (A2)
class NewFilterInit:
    """Gancho ``new_filters`` do ``widen`` (A3): filtros NOVOS por ``KernelInitializer.fit`` (A2) sobre o prefixo
    do PAI (canais herdados). Colunas de entrada novas dos filtros novos = 0; o ``widen`` zera as colunas
    do consumidor (saída nova = 0) -> função preservada. Conv não densa / falha -> init PyTorch, registrada."""

    def __init__(self, gev: GrowthEvaluator, method: str, method_cfg: dict, ipc: int, seed: int):
        from ga_search.abstract_init.base import get_initializer
        from ga_search.abstract_init.run_ablation import _Dispatch
        from ga_search.modehb.run_search import spifil_subset
        self.gev, self.method, self.seed = gev, method, int(seed)
        self.init = get_initializer(method, **dict(method_cfg or {}))
        k = min(int(ipc), int(gev.ev.spifil_k_max))
        self.train_data = _Dispatch(gev.ev, self.init).train_data(spifil_subset(gev.ev.spifil_pool, k))
        self.log: List[dict] = []
        self.spifil_over: Dict[str, Any] = {}          # growth v3: gene spifil_n_superpixels
        self.ipc_effective = int(k)
        try:
            self.n_images = int(len(self.train_data[1])) if isinstance(self.train_data, (tuple, list)) else None
        except Exception:
            self.n_images = None

    def scfg(self, seed: int) -> dict:
        return {**self.gev.spifil_cfg(seed), **self.spifil_over}

    def hook(self, parent: BB.Backbone, widened_prev: Dict[int, bool]):
        def fn(j: int, cb: nn.Module, out_local: List[int]):
            from ga_search.abstract_init.base import InitContext
            from ga_search.modehb.spifil_ops import to_ops_spec
            new_rows = [i for i, m in enumerate(out_local) if m < 0]
            conv = cb.conv
            rec = {"layer": int(j), "n_new": len(new_rows), "method": self.method, "n_fit": 0, "fallback": None}
            if isinstance(conv, nn.Sequential) and getattr(self, "dwsep_spifil", False) and len(conv) == 2 \
                    and isinstance(conv[1], nn.Conv2d) and conv[1].groups == 1 and conv[1].kernel_size == (1, 1):
                # v3.1 bloco 6: dwsep -> SPiFiL no POINTWISE novo, sobre o prefixo + o depthwise herdado do bloco
                pre = [copy.deepcopy(b) for b in parent.blocks[:j]] + [copy.deepcopy(parent.blocks[j].conv[0])]
                cin_par = _block_in(parent.spec, j)
                self._fit_rows(parent, j, conv[1], new_rows, cin_par, 1, pre, rec, tag="dwsep_pointwise")
                return
            if not (isinstance(conv, nn.Conv2d) and conv.groups == 1):
                rec["fallback"] = f"conv {type(conv).__name__}/groups={getattr(conv, 'groups', '?')}: init PyTorch"
                self.log.append(rec)
                return
            if j > 0 and widened_prev.get(j - 1) and parent.spec.layers()[j - 1].conv == "inception":
                rec["fallback"] = "entrada intercalada (inception alargado): init PyTorch"
                self.log.append(rec)
                return
            L = parent.spec.layers()[j]
            cin_par = _block_in(parent.spec, j)
            prefix = nn.Sequential(*[copy.deepcopy(b) for b in parent.blocks[:j]]).to(self.gev.device).eval() \
                if j > 0 else None
            if prefix is not None:
                prefix.out_channels = cin_par
            spec = to_ops_spec({"conv": "conv" if int(L.kernel) > 1 else "pointwise", "kernel": int(L.kernel),
                                "channels": len(new_rows), "activation": L.activation, "norm": "none",
                                "pool": "none", "residual": False, "dropout": 0.0}, cin_par)
            ctx = InitContext(seed=self.seed + j, device=self.gev.device, layer_idx=int(j),
                              prefix_id=f"growth:{parent.spec.hash12}:{j}", config=dict(self.init.config),
                              spifil_config=self.scfg(self.seed + j), genome=None,
                              diag_batch=self.gev.ev.svm_train_x[-16:])
            t0 = time.perf_counter()
            try:
                w, b, _prov = self.init.fit(prefix, self.train_data, spec, ctx)
                W, B = w["conv.weight"].detach().to(conv.weight), w.get("conv.bias", b.get("conv.bias"))
            except Exception as exc:          # InfeasibleCandidate, orçamento, ... -> registrado, init PyTorch
                rec["fallback"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                self.log.append(rec)
                return
            finally:                          # growth_mem: a cópia do prefixo do pai não fica na GPU
                del prefix
                _free(self.gev.device)
            n = min(int(W.shape[0]), len(new_rows))
            rows = torch.as_tensor(new_rows[:n], dtype=torch.long)
            with torch.no_grad():
                conv.weight[rows] = 0.0
                kc, kw = conv.weight.shape[-1], W.shape[-1]
                o = (kc - kw) // 2
                conv.weight[rows, :cin_par, o:o + kw, o:o + kw] = W[:n, :cin_par]
                if conv.bias is not None and B is not None:
                    conv.bias[rows] = B[:n].detach().to(conv.bias)
            rec.update(n_fit=n, n_pytorch=len(new_rows) - n, fit_seconds=time.perf_counter() - t0)
            self.log.append(rec)
        return fn


    def _fit_rows(self, parent, j: int, conv: nn.Conv2d, rows: List[int], cin_par: int, k: int, prefix_mods, rec: dict,
                  tag: str = "") -> None:
        """v3.1: SPiFiL (A2) nas linhas ``rows`` de ``conv`` com um prefixo explícito (ex.: dwsep pointwise)."""
        from ga_search.abstract_init.base import InitContext
        from ga_search.modehb.spifil_ops import to_ops_spec
        prefix = nn.Sequential(*prefix_mods).to(self.gev.device).eval() if prefix_mods else None
        if prefix is not None:
            prefix.out_channels = cin_par
        L = parent.spec.layers()[min(int(j), len(parent.spec.layers()) - 1)]
        spec = to_ops_spec({"conv": "conv" if k > 1 else "pointwise", "kernel": int(k), "channels": len(rows),
                            "activation": L.activation, "norm": "none", "pool": "none", "residual": False,
                            "dropout": 0.0}, cin_par)
        ctx = InitContext(seed=self.seed + 31 + j, device=self.gev.device, layer_idx=int(j),
                          prefix_id=f"growth_v31:{tag}:{parent.spec.hash12}:{j}", config=dict(self.init.config),
                          spifil_config=self.scfg(self.seed + 31 + j), genome=None,
                          diag_batch=self.gev.ev.svm_train_x[-16:])
        t0 = time.perf_counter()
        rec["target"] = tag
        try:
            w, b, _prov = self.init.fit(prefix, self.train_data, spec, ctx)
            W, B = w["conv.weight"].detach().to(conv.weight), w.get("conv.bias", b.get("conv.bias"))
        except Exception as exc:
            rec["fallback"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            self.log.append(rec)
            return
        finally:
            del prefix
            _free(self.gev.device)
        n = min(int(W.shape[0]), len(rows))
        with torch.no_grad():
            r = torch.as_tensor(list(rows[:n]), dtype=torch.long)
            conv.weight[r] = 0.0
            conv.weight[r, :cin_par] = W[:n, :cin_par].to(conv.weight)
            if conv.bias is not None and B is not None:
                conv.bias[r] = B[:n].detach().to(conv.bias)
        rec.update(n_fit=n, n_pytorch=len(rows) - n, fit_seconds=time.perf_counter() - t0)
        self.log.append(rec)

    # ---- growth v3: SPiFiL em linhas arbitrárias de uma conv densa (ramo novo do inception, filtros trocados)
    def conv_hook(self, parent: BB.Backbone, residual_of: Optional[Sequence[int]] = None, tag: str = "branch"):
        """-> fn(j, conv, rows): filtros SPiFiL (A2) ajustados sobre o prefixo do ``parent`` (canais herdados),
        escritos nas linhas ``rows`` de ``conv`` (nn.Conv2d densa; kernel = o da conv, centrado).
        ``residual_of`` (replace_bad_filters): linhas MANTIDAS da conv; os filtros novos são projetados no
        complemento ortogonal do espaço delas (no espaço de pesos) e re-escalados à norma original: ajustados
        no RESÍDUO do que os filtros mantidos já cobrem."""
        def fn(j: int, conv: nn.Module, rows: List[int]):
            from ga_search.abstract_init.base import InitContext
            from ga_search.modehb.spifil_ops import to_ops_spec
            rec = {"layer": int(j), "n_new": len(rows), "method": self.method, "n_fit": 0, "fallback": None,
                   "target": tag}
            if not (isinstance(conv, nn.Conv2d) and conv.groups == 1) or not rows:
                rec["fallback"] = "conv não densa: init PyTorch"
                self.log.append(rec)
                return
            cin_par = _block_in(parent.spec, j)
            k = int(conv.kernel_size[0])
            prefix = nn.Sequential(*[copy.deepcopy(b) for b in parent.blocks[:j]]).to(self.gev.device).eval() \
                if j > 0 else None
            if prefix is not None:
                prefix.out_channels = cin_par
            Ls = parent.spec.layers()
            L = Ls[min(int(j), len(Ls) - 1)]          # inserção no fim (j = n): ativação do último bloco
            spec = to_ops_spec({"conv": "conv" if k > 1 else "pointwise", "kernel": k, "channels": len(rows),
                                "activation": L.activation, "norm": "none", "pool": "none", "residual": False,
                                "dropout": 0.0}, cin_par)
            ctx = InitContext(seed=self.seed + 7919 + j, device=self.gev.device, layer_idx=int(j),
                              prefix_id=f"growth_v3:{tag}:{parent.spec.hash12}:{j}", config=dict(self.init.config),
                              spifil_config=self.scfg(self.seed + 7919 + j), genome=None,
                              diag_batch=self.gev.ev.svm_train_x[-16:])
            t0 = time.perf_counter()
            try:
                w, b, _prov = self.init.fit(prefix, self.train_data, spec, ctx)
                W, B = w["conv.weight"].detach().to(conv.weight), w.get("conv.bias", b.get("conv.bias"))
            except Exception as exc:
                rec["fallback"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                self.log.append(rec)
                return
            finally:
                del prefix
                _free(self.gev.device)
            n = min(int(W.shape[0]), len(rows))
            cin_c = min(cin_par, conv.weight.shape[1])
            kc, kw = conv.weight.shape[-1], W.shape[-1]
            Wn = torch.zeros(n, conv.weight.shape[1], kc, kc, dtype=conv.weight.dtype)
            o = (kc - kw) // 2
            if o >= 0:
                Wn[:, :cin_c, o:o + kw, o:o + kw] = W[:n, :cin_c].cpu()
            else:
                Wn[:, :cin_c] = W[:n, :cin_c, -o:-o + kc, -o:-o + kc].cpu()
            if residual_of:
                keep = conv.weight.detach()[torch.as_tensor(list(residual_of), dtype=torch.long)].reshape(len(residual_of), -1).double().cpu()
                Q, _ = torch.linalg.qr(keep.T)                     # base do espaço dos filtros mantidos
                F_ = Wn.reshape(n, -1).double()
                nrm = F_.norm(dim=1, keepdim=True)
                R = F_ - (F_ @ Q) @ Q.T
                R = R / R.norm(dim=1, keepdim=True).clamp_min(1e-12) * nrm
                Wn = R.reshape(Wn.shape).to(Wn.dtype)
                rec["residual_projection"] = {"n_keep": len(residual_of), "rank_keep": int(Q.shape[1])}
            with torch.no_grad():
                r = torch.as_tensor(list(rows[:n]), dtype=torch.long)
                conv.weight[r] = Wn.to(conv.weight)
                if conv.bias is not None and B is not None:
                    conv.bias[r] = B[:n].detach().to(conv.bias)
            rec.update(n_fit=n, n_pytorch=len(rows) - n, fit_seconds=time.perf_counter() - t0)
            self.log.append(rec)
        return fn


def _block_in(spec, j: int) -> int:
    """Canais de entrada do bloco j (growth v3: conta o dense_out do bloco anterior quando existir)."""
    f = getattr(BB, "block_in_channels", None)
    if j >= len(spec.layers()):                    # growth v3.1: entrada de um bloco inserido no fim
        return _block_out(spec, len(spec.layers()) - 1)
    if f is not None:
        return int(f(spec, j))
    return int(spec.input.in_channels) if j == 0 else int(spec.layers()[j - 1].channels)


def _block_out(spec, j: int) -> int:
    f = getattr(BB, "block_out_channels", None)
    if f is not None:
        return int(f(spec, j))
    return int(spec.layers()[j].channels)


# ═════════════════════════════════════════════════════════════════ 4. propostas (plano de morfismos)
def _band_bounds(band: str) -> Tuple[int, int]:
    i = BANDS.index(band)
    return (0 if i == 0 else BUDGETS[BANDS[i - 1]]), BUDGETS[band]


def _tied_widths(spec: BackboneSpec, widths: Dict[int, int]) -> Dict[int, int]:
    """Propaga a largura para camadas que exigem cout == cin (residual identity/gated, depthwise)."""
    layers = spec.layers()
    out = dict(widths)
    cin = int(spec.input.in_channels)
    for j, L in enumerate(layers):
        if (L.residual in ("identity", "gated") or L.conv == "depthwise"
                or getattr(L, "cov_gate", "none") != "none") and j > 0:
            prev = out.get(j - 1, int(layers[j - 1].channels))
            if prev != int(L.channels) or j in out:
                out[j] = prev
        cin = out.get(j, int(L.channels))
    return out


def _params(res: MO.GrowthResult) -> int:
    return int(sum(p.numel() for p in res.child.parameters()))


def _apply(res: MO.GrowthResult, op: dict, hook=None, fit_x=None) -> MO.GrowthResult:
    o = dict(op)
    name = o.pop("op")
    if name == "widen":
        w = {int(k): int(v) for k, v in o["widths"].items()}
        return MO.widen(res, w, new_filters=hook, new_init_label=o.get("new_init", "random:pytorch_default"),
                        seed=int(o.get("seed", 0)))
    if name == "deepen_identity":
        return MO.deepen_identity(res, int(o["at"]), kernel=int(o.get("kernel", 3)))
    if name == "deepen_residual_zero":
        kw = {"norm": o["norm"]} if o.get("norm", "none") != "none" else {}   # growth v3 (4d): whitening
        return MO.deepen_residual_zero(res, int(o["at"]), act=o.get("act", "silu"), kernel=int(o.get("kernel", 3)), **kw)
    if name == "add_skip_zero":
        return MO.add_skip_zero(res, int(o["layer"]))
    if name == "add_dense_layer":                  # growth v3.1 bloco 3: camada DenseNet (cat(H(x), x))
        from ga_search.backbones import morphisms_v3 as MV3
        return MV3.add_dense_layer(res, int(o["at"]), int(o["growth"]), kernel=int(o.get("kernel", 3)),
                                   norm=str(o.get("norm", "batch")), act=str(o.get("act", "relu")),
                                   new_filters=hook, seed=int(o.get("seed", 0)))
    if name in ("cov_gate", "deepen_rezero"):      # growth v3.1: residual gated (covariância / ReZero), g = 0
        from ga_search.backbones import morphisms_v3 as MV3
        return MV3.add_gated_residual(res, int(o["at"]), kernel=int(o.get("kernel", 3)),
                                      norm=str(o.get("norm", "zca" if name == "cov_gate" else "none")),
                                      act=str(o.get("act", "relu")),
                                      gate=str(o.get("gate", "channel" if name == "cov_gate" else "scalar")),
                                      tag=name, new_filters=hook, seed=int(o.get("seed", 0)), fit_x=fit_x)
    if name in ("inception_branch_add", "dense_concat", "replace_bad_filters"):
        from ga_search.backbones import morphisms_v3 as MV3
        if name == "inception_branch_add":       # growth v3 (4a)
            return MV3.inception_branch_add(res, int(o["layer"]), int(o["kernel"]), int(o["channels"]),
                                            new_filters=hook, seed=int(o.get("seed", 0)))
        if name == "dense_concat":               # growth v3 (4b)
            return MV3.dense_concat(res, int(o["layer"]))
        return res                               # replace_bad_filters: no planejamento é no-op (mesma arquitetura;
                                                 # as linhas vêm dos DADOS no realize_plan)
    raise ChainError(f"op {name!r}")


def propose_plan(parent: BB.Backbone, target_band: str, theta: Sequence[float], seed: int, gcfg: dict,
                 hard_cap: int = HARD_CAP) -> Optional[List[dict]]:
    """Plano (lista de ops) que leva o pai à faixa ``target_band`` só com morfismos exatos. Determinístico em
    (pai, theta, seed). theta = (p_widen, p_deepen, escala do fator, foco de camada) em [0,1]^4.
    Devolve None se não couber (teto duro sempre respeitado)."""
    lo, hi = _band_bounds(target_band)
    hi = min(hi, int(hard_cap))
    rng = np.random.default_rng(int(seed))
    t = [min(max(float(x), 0.0), 1.0) for x in theta]
    w = dict(gcfg.get("ops_weights") or GROWTH_DEFAULTS["ops_weights"])
    w["widen"] *= 0.5 + t[0]
    w["deepen_residual_zero"] *= 0.5 + t[1]
    w["deepen_identity"] *= 0.5 + t[1]
    names = list(w)
    probs = np.asarray([w[n] for n in names], dtype=float)
    probs /= probs.sum()
    res = MO.start(parent)
    ops: List[dict] = []
    for _ in range(int(gcfg.get("max_ops", 12)) * 3):
        n_now = _params(res)
        if lo < n_now <= hi:
            return ops
        if len(ops) >= int(gcfg.get("max_ops", 12)) or n_now > hi:
            return None
        spec = res.child.spec
        layers = spec.layers()
        n = len(layers)
        name = names[int(rng.choice(len(names), p=probs))]
        op = None
        if name == "widen":
            focus = int(round(t[3] * (n - 1)))
            k = int(rng.integers(1, n + 1))
            cand = sorted(set([focus] + [int(x) for x in rng.choice(n, size=min(k, n), replace=False)]))
            need = max(1.05, (math.sqrt(max(lo, 1) * hi) / max(n_now, 1)))
            fac = 1.0 + (min(need, 2.5) - 1.0) * (0.35 + 0.65 * t[2]) * float(rng.uniform(0.6, 1.0))
            widths = {j: max(int(layers[j].channels) + 1, int(round(int(layers[j].channels) * fac))) for j in cand}
            widths = _tied_widths(spec, widths)
            op = {"op": "widen", "widths": {str(j): int(v) for j, v in sorted(widths.items())},
                  "seed": int(rng.integers(0, 2 ** 31 - 1))}
        elif name == "deepen_identity":
            ok = [a for a in range(1, n + 1) if MO._nonneg_output(layers[a - 1])]
            if ok:
                op = {"op": "deepen_identity", "at": int(rng.choice(ok)), "kernel": 3}
        elif name == "deepen_residual_zero":
            op = {"op": "deepen_residual_zero", "at": int(rng.integers(1, n + 1)),
                  "act": gcfg.get("residual_act", "silu"), "kernel": 3}
        elif name == "add_skip_zero":
            ok = [j for j, L in enumerate(layers) if L.residual == "none" and j > 0]
            if ok:
                op = {"op": "add_skip_zero", "layer": int(rng.choice(ok))}
        if op is None:
            continue
        for _try in range(4):                 # passou do teto: encolhe o incremento do widen
            try:
                nxt = _apply(res, op)
            except (MO.MorphismError, BB.BudgetError, ValueError):
                nxt = None
                break
            if _params(nxt) <= hi:
                break
            if op["op"] != "widen":
                nxt = None
                break
            widths = {int(j): int(layers[int(j)].channels) + max(1, (int(v) - int(layers[int(j)].channels)) // 2)
                      for j, v in op["widths"].items()}
            op["widths"] = {str(j): v for j, v in sorted(_tied_widths(spec, widths).items())}
            nxt = None
        if nxt is None or _params(nxt) > hi:
            continue
        res = nxt
        ops.append(op)
    n_now = _params(res)
    return ops if lo < n_now <= hi else None


# ---- growth_fix4 (c): planner por família de operação (variedade + widen encadeado até a faixa)
_PLAN_ERRORS = (MO.MorphismError, BB.BudgetError, ValueError, KeyError, IndexError, RuntimeError, NotImplementedError)


def _try_apply(res: MO.GrowthResult, op: dict) -> Optional[MO.GrowthResult]:
    try:
        return _apply(res, op)
    except _PLAN_ERRORS:
        return None


def _widen_op(res: MO.GrowthResult, idx: Sequence[int], f: float, seed: int) -> dict:
    spec = res.child.spec
    layers = spec.layers()
    idx = sorted({int(j) for j in idx})
    while True:          # camada amarrada (residual identity/gated, depthwise): alarga junto o bloco anterior
        widths = {j: max(int(layers[j].channels) + 1, int(round(int(layers[j].channels) * float(f)))) for j in idx}
        widths = _tied_widths(spec, widths)
        stuck = [j for j in idx if widths.get(j, int(layers[j].channels)) <= int(layers[j].channels)]
        if not stuck or min(stuck) == 0 or (min(stuck) - 1) in idx:
            break
        idx = sorted(set(idx) | {min(stuck) - 1})
    return {"op": "widen", "widths": {str(j): int(v) for j, v in sorted(widths.items())}, "seed": int(seed)}


def _widen_toward(res: MO.GrowthResult, idx: Sequence[int], target: int, lo: int, hi: int, fmax: float,
                  seed: int, iters: int = 14):
    """UM widen dos blocos ``idx`` (fator em (1, fmax]) que leva params o mais perto possível de ``target`` sem
    passar de ``hi``. Devolve (op, res) — possivelmente ainda abaixo da faixa (o chamador encadeia) — ou
    (None, None) se nem o incremento mínimo (+1 canal) cabe/aplica."""
    n0 = _params(res)
    op = _widen_op(res, idx, fmax, seed)
    r = _try_apply(res, op)
    if r is not None and n0 < _params(r) <= target:
        return op, r
    a, b = 1.0, float(fmax)
    below = above = None
    for _ in range(int(iters)):
        f = 0.5 * (a + b)
        op = _widen_op(res, idx, f, seed)
        r = _try_apply(res, op)
        if r is None or _params(r) <= _params(res):
            b = f
            continue
        p = _params(r)
        if p <= target:
            a, below = f, (op, r)
        else:
            b = f
            if p <= hi:
                above = (op, r)
    if below is not None and lo < _params(below[1]) <= hi:
        return below
    if above is not None:
        return above
    if below is not None:
        return below
    op = _widen_op(res, idx, 1.0, seed)        # incremento mínimo: +1 canal
    r = _try_apply(res, op)
    if r is not None and n0 < _params(r) <= hi:
        return op, r
    return None, None


V3_OPS = ("inception_branch_add", "dense_concat", "replace_bad_filters", "cov_gate", "deepen_rezero", "add_dense_layer")


def plan_families(parent: BB.Backbone, gcfg: Optional[dict] = None) -> List[Tuple[str, Any]]:
    """Famílias de 1ª operação, intercaladas por tipo: widen do bloco j (j < último), head_widen (widen do
    último bloco, que alarga a entrada da cabeça), deepen_residual_zero, add_skip_zero, deepen_identity.
    growth v3: + inception_branch_add / dense_concat / replace_bad_filters quando em ``v3.ops``; ``v3.families``
    (lista de tipos) restringe às famílias do braço."""
    layers = parent.spec.layers()
    n = len(layers)
    groups = [[("widen", j) for j in range(n - 1)],
              [("head_widen", n - 1)] + ([("widen_all", None)] if n > 1 else []),
              [("deepen_residual_zero", a) for a in range(n, 0, -1)],
              [("add_skip_zero", j) for j, L in enumerate(layers) if L.residual == "none" and j > 0],
              [("deepen_identity", a) for a in range(1, n + 1) if MO._nonneg_output(layers[a - 1])]]
    v3 = _v3(gcfg or {})
    ops = [o for o in (v3.get("ops") or []) if o in V3_OPS]
    dense_ok = lambda j: not getattr(layers[j], "dense_out", False)                   # noqa: E731
    conv_ok = lambda L: L.conv in ("conv", "pointwise", "inception") and L.residual not in ("identity", "gated")  # noqa
    if "inception_branch_add" in ops:
        groups.append([("inception_branch_add", j) for j in range(n - 1, -1, -1) if conv_ok(layers[j])])
    if "dense_concat" in ops:
        groups.append([("dense_concat", j) for j in range(n - 2, -1, -1) if dense_ok(j)
                       and layers[j + 1].residual not in ("identity", "gated") and layers[j + 1].conv != "depthwise"])
    if "replace_bad_filters" in ops:
        groups.append([("replace_bad_filters", j) for j in range(n) if layers[j].conv in ("conv", "pointwise")])
    if "add_dense_layer" in ops:                        # v3.1 bloco 3: consumidor sem amarra cin == cout
        groups.append([("add_dense_layer", a) for a in range(n - 1, 0, -1)
                       if (layers[a].residual not in ("identity", "gated") and layers[a].conv != "depthwise"
                                     and getattr(layers[a], "cov_gate", "none") == "none")])
    for kind in ("cov_gate", "deepen_rezero"):           # growth v3.1: residual gated, inserido antes de a
        if kind in ops:
            groups.append([(kind, a) for a in range(n, 0, -1)])
    fams = v3.get("families")
    if fams:
        allowed = set(fams)
        groups = [[f for f in g if f[0] in allowed or (f[0] in ("head_widen", "widen_all") and "widen" in allowed)
                   or (f[0] == "deepen_residual_zero" and "residual" in allowed)] for g in groups]
        groups = [g for g in groups if g] or [[]]
    out: List[Tuple[str, Any]] = []
    for k in range(max(len(g) for g in groups)):
        out += [g[k] for g in groups if k < len(g)]
    return out


def family_label(fam: Tuple[str, Any]) -> str:
    kind, j = fam
    return {"widen": f"widen_b{j}", "head_widen": "head_widen", "widen_all": "widen_all", "deepen_residual_zero": f"deepen_residual_zero@{j}",
            "add_skip_zero": f"add_skip_zero@{j}", "deepen_identity": f"deepen_identity@{j}",
            "inception_branch_add": f"inception_branch_add@{j}", "dense_concat": f"dense_concat@{j}",
            "replace_bad_filters": f"replace_bad_filters@{j}", "cov_gate": f"cov_gate@{j}",
            "deepen_rezero": f"deepen_rezero@{j}", "add_dense_layer": f"add_dense_layer@{j}"}[kind]


def plan_with_family(parent: BB.Backbone, target_band: str, family: Tuple[str, Any], theta: Sequence[float],
                     seed: int, gcfg: dict, hard_cap: int = HARD_CAP) -> Tuple[Optional[List[dict]], Optional[str]]:
    """Plano exato que começa pela operação da ``family`` e completa a faixa com widens encadeados (fator
    <= ``max_widen_factor`` por op; pais tiny precisam de vários). Devolve (ops, None) ou (None, motivo)."""
    lo, hi = _band_bounds(target_band)
    hi = min(hi, int(hard_cap))
    t = [min(max(float(x), 0.0), 1.0) for x in (list(theta) + [0.5] * 4)[:4]]
    target = int(lo + (hi - lo) * (0.2 + 0.6 * t[2]))      # ponto-alvo dentro da faixa (theta[2])
    fmax = float(gcfg.get("max_widen_factor", 2.5))
    max_ops = int(gcfg.get("max_ops", 12))
    rng = np.random.default_rng(int(seed))
    sd = lambda: int(rng.integers(0, 2 ** 31 - 1))           # noqa: E731
    res = MO.start(parent)
    if _params(res) > hi:
        return None, f"pai com {_params(res)} params já acima da faixa {target_band} (<= {hi})"
    kind, j = family
    ops: List[dict] = []
    if kind in ("widen", "head_widen", "widen_all"):
        idx0 = list(range(len(parent.spec.layers()))) if kind == "widen_all" else [int(j)]
        for _ in range(3):                                    # widen do bloco escolhido, encadeado se preciso
            op, r = _widen_toward(res, idx0, target, lo, hi, fmax, sd())
            if op is None:
                break
            res, ops = r, ops + [op]
            if lo < _params(res) <= hi or _params(res) >= target:
                break
        if not ops:
            return None, f"{family_label(family)}: nem +1 canal cabe/aplica"
    elif kind == "replace_bad_filters":
        pass                                                  # growth v3 (4f): widens primeiro, replace no FIM
    else:
        v3 = _v3(gcfg)
        L = parent.spec.layers()[int(j)] if kind in ("inception_branch_add", "dense_concat") else None
        bk = [int(k) for k in (v3.get("branch_kernels") or [1, 3, 5])]
        op = {"deepen_residual_zero": {"op": "deepen_residual_zero", "at": int(j),
                                       "act": gcfg.get("residual_act", "silu"), "kernel": 3,
                                       **({"norm": str(v3["whiten_new"])} if v3.get("whiten_new") not in
                                          (None, "none") else {})},
              "deepen_identity": {"op": "deepen_identity", "at": int(j), "kernel": 3},
              "add_skip_zero": {"op": "add_skip_zero", "layer": int(j)},
              "inception_branch_add": ({"op": "inception_branch_add", "layer": int(j), "kernel": bk[int(rng.integers(len(bk)))],
                                        "channels": max(4, int(round(float(v3.get("branch_frac", 0.25)) * int(L.channels)))),
                                        "seed": sd()} if L is not None else None),
              "dense_concat": {"op": "dense_concat", "layer": int(j)},
              "cov_gate": {"op": "cov_gate", "at": int(j), "kernel": 3, "gate": str(v3.get("cov_gate_mode", "channel")),
                           "norm": str(v3["whiten_new"]) if v3.get("whiten_new") not in (None, "none") else "zca",
                           "act": "relu", "seed": sd()},
              "deepen_rezero": {"op": "deepen_rezero", "at": int(j), "kernel": 3, "gate": "scalar", "norm": "none",
                                "act": "relu", "seed": sd()},
              "add_dense_layer": {"op": "add_dense_layer", "at": int(j), "kernel": 3, "norm": "batch", "act": "relu",
                                  "growth": max(4, int(round(float(v3.get("dense_growth_frac", 0.25))
                                                             * int(parent.spec.block_out_channels(int(j) - 1))))),
                                  "seed": sd()} if kind == "add_dense_layer" else None}[kind]
        r = _try_apply(res, op)
        if r is None:
            return None, f"{family_label(family)}: morfismo inaplicável"
        if _params(r) > hi:
            return None, f"{family_label(family)}: {_params(r)} params passa do teto da faixa {hi}"
        res, ops = r, [op]
    while not (lo < _params(res) <= hi):                     # completa com widens encadeados
        if len(ops) >= max_ops:
            return None, f"{family_label(family)}: {_params(res)} params após {max_ops} ops (faixa ({lo}, {hi}])"
        n = len(res.child.spec.layers())
        step = None
        for idx in [list(range(n)), [n - 1]] + [[k] for k in range(n - 1)]:
            op, r = _widen_toward(res, idx, target, lo, hi, fmax, sd())
            if op is not None and _params(r) > _params(res):
                step = (op, r)
                break
        if step is None:
            return None, f"{family_label(family)}: nenhum widen cabe a partir de {_params(res)} params"
        res, ops = step[1], ops + [step[0]]
    if kind == "replace_bad_filters":
        ops = ops + [{"op": "replace_bad_filters", "layer": int(j)}]
    return ops, None


def _plan_shape(ops: Sequence[dict]) -> str:
    """Forma do plano (op + onde; sem os valores das larguras): dois planos com a mesma forma no mesmo pai
    contam como repetidos (ex.: widen_b0 e head_widen amarrados por residual identity)."""
    return json.dumps([[o["op"], sorted(o["widths"]) if o["op"] == "widen" else o.get("at", o.get("layer"))]
                       for o in ops])


def plan_children(parents: List[Tuple[dict, BB.Backbone]], target_band: str, n_children: int,
                  thetas: Sequence[Sequence[float]], seeds: Sequence[int], gcfg: dict,
                  hard_cap: int = HARD_CAP, mem_ctx: Optional[dict] = None,
                  gcfgs: Optional[Sequence[dict]] = None) -> List[dict]:
    """n_children planos válidos, pais em round-robin; o k-ésimo filho de um pai começa pela k-ésima família
    (widen em blocos diferentes, head_widen, deepen, add_skip...). Plano repetido do mesmo pai é pulado; pai
    esgotado cede a vaga ao próximo pai. Cada item: {parent, ops, family, theta, seed} ou {parent, ops=None,
    reason} quando NENHUM pai/família produz plano (motivos agregados).

    growth_fix5: com ``mem_ctx`` (``mem_ctx_from``) e ``max_gb`` definido, plano com pico estimado acima do teto
    vira "infeasible_mem" (registrado em ``infeasible_mem`` do item que o substitui) e a próxima família entra."""
    P = len(parents)
    used: Dict[int, set] = {p: set() for p in range(P)}
    nxt = {p: 0 for p in range(P)}
    fams = {p: plan_families(parents[p][1], gcfg) for p in range(P)}
    reasons: Dict[int, List[str]] = {p: [] for p in range(P)}
    out: List[dict] = []
    for i in range(int(n_children)):
        got = None
        pending_inf: List[dict] = []
        for d in range(P):
            p = (i + d) % P
            pm = parents[p][1]
            while nxt[p] < len(fams[p]) and got is None:
                fam = fams[p][nxt[p]]
                nxt[p] += 1
                ops, why = plan_with_family(pm, target_band, fam, thetas[i], seeds[i],
                                            gcfgs[i] if gcfgs is not None else gcfg, hard_cap)
                if ops is None:
                    reasons[p].append(why)
                    continue
                key = _plan_shape(ops)
                if key in used[p]:
                    reasons[p].append(f"{family_label(fam)}: plano repetido")
                    continue
                used[p].add(key)
                est = None
                if mem_ctx is not None and mem_ctx.get("max_gb") is not None:
                    est = plan_memory(pm, ops, mem_ctx)
                    if not est["feasible"]:
                        why = (f"{family_label(fam)}: infeasible_mem (pico estimado {est['peak_gb']:.1f} GB "
                               f"[ssl {est['ssl_gb']:.1f}, ae {est['ae_gb']:.1f}] > {float(mem_ctx['max_gb']):.1f} GB)")
                        reasons[p].append(why)
                        pending_inf.append({"parent": parents[p][0], "ops": ops, "family": family_label(fam),
                                            "theta": list(thetas[i]), "seed": int(seeds[i]), "reason": why,
                                            "mem_estimate": est})
                        continue
                got = {"parent": parents[p][0], "ops": ops, "family": family_label(fam), "theta": list(thetas[i]),
                       "seed": int(seeds[i]), "parent_index": p}
                if est is not None:
                    got["mem_estimate"] = est
            if got is not None:
                break
        if got is None:
            why = "; ".join(f"{parents[p][0].get('node_id')}: " + " | ".join(reasons[p][-6:]) for p in range(P)) \
                or "sem pais"
            got = {"parent": parents[i % P][0] if P else None, "ops": None, "theta": list(thetas[i]),
                   "seed": int(seeds[i]), "reason": f"sem plano na faixa (todas as famílias esgotadas): {why}"}
        got["infeasible_mem"] = pending_inf
        out.append(got)
    return out


def realize_plan(parent: BB.Backbone, ops: Sequence[dict], nfi: Optional[NewFilterInit],
                 gcfg: Optional[dict] = None) -> MO.GrowthResult:
    """Reaplica o plano com SPiFiL nos canais novos (o plano foi achado com init PyTorch; exatidão igual).

    growth v3: ramo novo do inception com SPiFiL (``conv_hook``); SoftHebb opcional nos filtros NOVOS (antes do
    AE/SSL); ``replace_bad_filters`` (sempre no FIM do plano) usa as respostas do bloco nas imagens de TREINO.
    Com replace, ``res.v3_pre_replace`` = o resultado exato ANTES da troca (âncora do WiSE = o pai em função) e
    ``res.v3_replace`` = o registro."""
    v3 = _v3(gcfg or {})
    res = MO.start(parent)
    exact_ops = [o for o in ops if o["op"] != "replace_bad_filters"]
    rep_ops = [o for o in ops if o["op"] == "replace_bad_filters"]
    for op in exact_ops:
        hook = None
        if op["op"] == "widen" and nfi is not None:
            cur = res.child
            widened = {int(j): True for j, v in op["widths"].items() if int(v) != int(cur.spec.layers()[int(j)].channels)}
            hook = nfi.hook(cur, widened)
            op = {**op, "new_init": f"{nfi.method}:residual_zero_out"}
        elif op["op"] == "inception_branch_add" and nfi is not None:
            hook = nfi.conv_hook(res.child, tag="inception_branch")
        elif op["op"] == "add_dense_layer" and nfi is not None:                   # v3.1: SPiFiL na camada densa
            hook = nfi.conv_hook(res.child, tag="add_dense_layer")
        elif op["op"] in ("cov_gate", "deepen_rezero") and nfi is not None:      # growth v3.1: SPiFiL na conv do ramo
            hook = nfi.conv_hook(res.child, tag=op["op"])
            res = _apply(res, op, hook, fit_x=nfi.gev.ev.svm_train_x[: int(v3.get("cov_gate_fit_images", 64))])
            continue
        res = _apply(res, op, hook)
    prune_log = []
    if v3.get("prune_redundant_new") and nfi is not None:      # v3.1 bloco 5: poda exata de canais NOVOS redundantes
        res, prune_log = prune_redundant_new(res, nfi.gev, v3, gcfg or {})
    sh_log = []
    shc = {**(v3.get("softhebb_cfg") or {}), "paper": bool(v3.get("softhebb_paper")),
           "radius": float(v3.get("softhebb_radius", 3.0))}
    if v3.get("softhebb") and nfi is not None:
        sh_log += softhebb_new_filters(res, nfi.gev, shc, seed=nfi.seed)
    if rep_ops:
        if nfi is None:                      # planejamento / estimador: a arquitetura não muda
            return res
        pre = copy.deepcopy(res)
        infos = []
        for op in rep_ops:
            res, info = replace_bad_filters_data(res, int(op["layer"]), nfi, v3.get("replace") or {})
            infos.append(info)
        if v3.get("softhebb"):
            sh_log += softhebb_new_filters(res, nfi.gev, shc, seed=nfi.seed + 1, skip_from=pre)
        if any(i.get("n_replaced", 0) for i in infos):
            res.v3_pre_replace = pre
        res.v3_replace = infos
    res.v3_softhebb = sh_log
    res.v3_prune = prune_log
    return res


def prune_redundant_new(res: MO.GrowthResult, gev: GrowthEvaluator, v3: dict, gcfg: dict):
    """v3.1 (pedido do usuário: mais superpixels -> mais filtros; a redundância sai DENTRO da rede): entre os canais
    NOVOS de cada bloco, os mortos (variância ~0) e os quase duplicados (|corr| > thr com qualquer canal) nas respostas
    de TREINO são PODADOS (exato: colunas 0 no consumidor). Nunca leva o filho abaixo da faixa-alvo (poda parcial)."""
    from ga_search.backbones import morphisms_v3 as MV3
    pc = {**{"corr_thr": 0.95, "var_rel_eps": 1e-6, "n_images": 64}, **(v3.get("prune") or {})}
    lo = _band_bounds(band_of(_params(res)))[0]        # a poda nunca tira o filho da faixa que o plano atingiu
    log = []
    x = gev.ev.svm_train_x[: int(pc["n_images"])]
    for j in range(len(res.child.blocks)):
        om = res.out_maps[j]
        if om is None or not any(m < 0 for m in om):
            continue
        L = res.child.spec.layers()[j]
        C = int(L.channels)
        acts = _block_acts(res.child, j, x, gev.device)
        bad = MV3.bad_filters(acts, corr_thr=float(pc["corr_thr"]), var_rel_eps=float(pc["var_rel_eps"]), n_conv=C)
        cand = sorted({int(c) for c in bad["dead"]} | {int(d[0]) for d in bad["dup"]})
        rows = [c for c in cand if c < len(om) and om[c] < 0]
        rec = {"block": j, "n_new": sum(1 for m in om if m < 0), "n_candidates": len(rows), "n_pruned": 0}
        while rows:
            try:
                r2 = MV3.prune_new_channels(res, j, rows)
            except MO.MorphismError as exc:
                rec["skip"] = str(exc)[:160]
                break
            if _params(r2) > lo:
                res = r2
                rec["n_pruned"] = len(rows)
                break
            rows = rows[: len(rows) // 2]          # não sai da faixa: poda menos
        log.append(rec)
    return res, log


@torch.no_grad()
def _block_acts(model: BB.Backbone, j: int, x: torch.Tensor, device, bs: int = 32) -> torch.Tensor:
    model.to(device).eval()
    outs = []
    for i0 in range(0, len(x), bs):
        h = x[i0:i0 + bs].to(device).float()
        for b in model.blocks[: j + 1]:
            h = b(h)
        outs.append(h.cpu())
    model.cpu()
    return torch.cat(outs)


def replace_bad_filters_data(res: MO.GrowthResult, j: int, nfi: NewFilterInit, rcfg: dict):
    """growth v3 (4f): acha filtros mortos (variância ~0) ou quase duplicados (|corr| > thr) nas RESPOSTAS do bloco
    j (imagens de TREINO) e os troca por filtros SPiFiL ajustados no resíduo (complemento dos mantidos). NÃO
    preserva a função: quem decide é o WiSE com âncora = pai (alpha=0 = pai) e kappa >= pai."""
    from ga_search.backbones import morphisms_v3 as MV3
    n_img = int(rcfg.get("n_images", 64))
    x = nfi.gev.ev.svm_train_x[:n_img]
    acts = _block_acts(res.child, j, x, nfi.gev.device)
    C = int(res.child.spec.layers()[j].channels)
    bad = MV3.bad_filters(acts, corr_thr=float(rcfg.get("corr_thr", 0.95)),
                          var_rel_eps=float(rcfg.get("var_rel_eps", 1e-6)), n_conv=C)   # dense: só a conv
    crit = str(rcfg.get("criterion", "corr"))
    cap = max(1, int(float(rcfg.get("max_frac", 0.25)) * C))
    if crit == "redo":                        # v3.1 (redo_repr.md): tau-dormentes, saída pós-ativação, treino
        dead = MV3.redo_dormant(acts, float(rcfg.get("redo_tau", 0.1)), n_conv=C)
        dups = []
    elif crit == "repr":                      # v3.1: RePr, os p=30% mais redundantes (O_f) dos filtros
        conv = res.child.blocks[j].conv
        Wc = conv.weight if isinstance(conv, nn.Conv2d) else conv[-1].weight
        dead, dups = MV3.repr_select(Wc, float(rcfg.get("repr_p", 0.30))), []
        cap = len(dead)
    else:
        dead = [int(c) for c in bad["dead"]]
        dups = [d for d in bad["dup"] if int(d[0]) not in dead]
    rows = (dead + [int(d[0]) for d in dups])[:cap]
    if crit in ("redo", "repr"):              # saída zerada (sem merge), como no artigo
        bad = {**bad, "mean": {}}
    merge = {int(d[0]): (int(d[1]), float(d[2]), float(d[3])) for d in dups if int(d[0]) in rows}
    means = bad.get("mean") or {}
    for c in dead:                            # canal morto ~ constante: entra só o termo b (= média) no consumidor
        if c in rows and (c in means or str(c) in means):
            merge[c] = (-1, 0.0, float(means.get(c, means.get(str(c)))))
    info = {"layer": int(j), "n_images": int(x.shape[0]), "split": "train", "criterion": crit,
            "n_dead": len(dead), "n_dup": len(dups),
            "rows": rows, "merge": {str(k): v for k, v in merge.items()}, "cap": cap,
            "corr_thr": float(rcfg.get("corr_thr", 0.95)), "n_replaced": len(rows)}
    if not rows:
        info["note"] = "nenhum filtro ruim: replace vira no-op"
        return res, info
    keep = [c for c in range(C) if c not in set(rows)]
    hook = nfi.conv_hook(res.child, residual_of=keep, tag="replace")
    out = MV3.replace_filters(res, j, rows, merge=merge, new_filters=hook, seed=nfi.seed + 104729 + j)
    return out, info


def softhebb_new_filters(res: MO.GrowthResult, gev: GrowthEvaluator, scfg: dict, seed: int = 0,
                         skip_from: Optional[MO.GrowthResult] = None) -> List[dict]:
    """growth v3 (4e): refinamento SoftHebb (Journé et al. 2023; ``modehb.softhebb.fit_softhebb``) SÓ nas linhas
    NOVAS e não nulas das convs densas (máscara toda False), sobre a entrada do bloco (prefixo congelado) nas
    imagens de TREINO, ANTES do AE/SSL. Convs zeradas (deepen_residual_zero) ficam intactas; as colunas do
    consumidor dos canais novos seguem 0 -> a preservação não muda (medida depois)."""
    from ga_search.modehb.softhebb import fit_softhebb
    model = res.child
    n_img = int(scfg.get("n_images", 128))
    x = gev.ev.svm_train_x[:n_img].float()
    log = []
    pre_sd = skip_from.child.state_dict() if skip_from is not None else None
    torch.manual_seed(int(seed))
    for j, blk in enumerate(model.blocks):
        for name, mod in blk.named_modules():
            if not (isinstance(mod, nn.Conv2d) and mod.groups == 1):
                continue
            key = f"blocks.{j}.{name}.weight"
            m = res.masks.get(key)
            if m is None:
                continue
            W = mod.weight.detach()
            rows = [r for r in range(W.shape[0]) if not bool(m[r].any()) and bool(W[r].abs().sum() > 0)]
            if pre_sd is not None and key in pre_sd and pre_sd[key].shape == W.shape:
                rows = [r for r in rows if not torch.equal(pre_sd[key][r], W[r])]
            if not rows:
                continue
            if name not in ("conv", "skip") and not name.startswith("conv.branches."):
                continue
            if name == "skip":
                continue
            tmp = nn.Conv2d(mod.in_channels, len(rows), mod.kernel_size, 1, mod.padding, bias=False)
            with torch.no_grad():
                tmp.weight.copy_(W[torch.as_tensor(rows)])
            prefix = nn.Sequential(*model.blocks[:j]).eval()
            pre_pool = None
            if name.startswith("conv.branches.") and isinstance(blk.conv.branches[name.split(".")[2]], nn.Sequential):
                continue
            paper = bool(scfg.get("paper"))
            if paper:                            # v3.1 (softhebb.md): tau/lr por camada, raio R, BN sem treino
                inv_tau = [1.0, 0.65, 0.25][min(j, 2)]
                lr_sh, temp = [0.08, 0.005, 0.01][min(j, 2)], 1.0 / inv_tau
                bs_sh = int(scfg.get("batch", 10))
                with torch.no_grad():
                    norms = tmp.weight.flatten(1).norm(dim=1).clamp_min(1e-12)
                    tmp.weight.mul_((float(scfg.get("radius", 3.0)) / norms).view(-1, 1, 1, 1))
                bn = lambda t: torch.nn.functional.batch_norm(t, None, None, training=True)   # noqa: E731
                tf = (lambda t, p=prefix: bn(p(t))) if j > 0 else bn
            else:
                lr_sh, temp, bs_sh = float(scfg.get("lr", 0.01)), float(scfg.get("temperature", 1.0)), 32
                tf = (lambda t, p=prefix: p(t)) if j > 0 else None
            try:
                info = fit_softhebb(tmp, [x[i:i + bs_sh] for i in range(0, len(x), bs_sh)], int(scfg.get("epochs", 1)),
                                    lr_sh, temp, "cpu", input_transform=tf)
                with torch.no_grad():
                    if paper:                    # devolve cada filtro à norma SPiFiL original (só a direção muda)
                        tmp.weight.mul_((norms / tmp.weight.flatten(1).norm(dim=1).clamp_min(1e-12)).view(-1, 1, 1, 1))
                    mod.weight[torch.as_tensor(rows)] = tmp.weight.detach().to(mod.weight)
                log.append({"param": key, "rows": len(rows), "status": "ok", "paper": paper, "lr": lr_sh,
                            "temperature": temp,
                            **{k: v for k, v in (info or {}).items() if isinstance(v, (int, float, str, bool))}})
            except Exception as exc:            # registrado; filtros SPiFiL ficam como estavam
                log.append({"param": key, "rows": len(rows), "status": f"falhou: {type(exc).__name__}: {str(exc)[:120]}"})
            del pre_pool
    return log


# ═════════════════════════════════════════════════════════════════ 5. treino de um filho
def ssl_disabled(gcfg: dict) -> bool:
    """``ssl: none`` (ou null/false): pula o SSL global; fica só o AE local nos blocos novos. Padrão: lejepa."""
    v = gcfg.get("ssl", "lejepa")
    return v is None or v is False or str(v).strip().lower() in ("none", "off", "false", "")


def _blend_into(model: BB.Backbone, anchor: dict, trained: dict, a: float) -> None:
    """WiSE: (1-a)*âncora + a*treinado. a == 0 carrega a ÂNCORA literalmente (bit a bit)."""
    if a == 0.0:
        model.load_state_dict({k: v.clone() for k, v in anchor.items()})
        return
    dev = {k: v.device for k, v in model.state_dict().items()}   # growth_mem: soma no device do modelo
    sd = {}
    for k, t in trained.items():
        a0 = anchor.get(k)
        if a0 is not None and t.is_floating_point() and a0.shape == t.shape:
            sd[k] = (1 - a) * a0.to(dev[k]) + a * t.to(dev[k])
        else:
            sd[k] = t
    model.load_state_dict(sd)
    del sd


def _masks_under(masks: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {k[len(prefix):]: v for k, v in masks.items() if k.startswith(prefix)}


def _finite_state(sd: dict) -> bool:
    return all(bool(torch.isfinite(v).all()) for v in sd.values() if v.is_floating_point())


def _snapper(mem: dict):
    return (lambda m: {k: v.detach().to("cpu", copy=True) for k, v in m.state_dict().items()}) \
        if mem["wise_on_cpu"] else (lambda m: {k: v.detach().clone() for k, v in m.state_dict().items()})


class CovGatedHead(nn.Module):
    """growth v3.1 (pedido do usuário): cabeça do SSL = [cabeça antiga (congelada, fora dos parâmetros),
    g_h * Cov(f)], Cov = triu da covariância por imagem de P^T f (P ortonormal fixa C x k, semeada), L2.
    g_h escalar treinável iniciado em 0: o SSL decide se usa a 2ª ordem. Só existe no SSL (como o projetor);
    a leitura da fitness continua a cabeça so2 fixa. ``gate_value`` vai para o result.json."""

    def __init__(self, old_head: nn.Module, cin: int, k: int = 16, seed: int = 0):
        super().__init__()
        self.__dict__["old"] = old_head
        g = torch.Generator().manual_seed(int(seed) + 7919 * int(cin))
        kk = min(int(k), int(cin))
        Q, _ = torch.linalg.qr(torch.randn(int(cin), kk, generator=g))
        self.register_buffer("P", Q.contiguous())
        self.g = nn.Parameter(torch.zeros(1))
        self.iu = torch.triu_indices(kk, kk)

    def cov(self, f):
        y = f.float().flatten(2).transpose(1, 2) @ self.P.to(f.device)        # [B, HW, k]
        y = y - y.mean(1, keepdim=True)
        c = y.transpose(1, 2) @ y / max(y.shape[1], 1)
        z = c[:, self.iu[0], self.iu[1]]
        return z / z.norm(dim=1, keepdim=True).clamp_min(1e-6)

    def forward(self, f):
        self.old.eval()
        e = self.old(f)
        return torch.cat([e, (self.g * self.cov(f)).to(e.dtype)], 1)


def _global_ssl(gev: GrowthEvaluator, model: BB.Backbone, masks: dict, scale: float, epochs: int, gcfg: dict,
                seed: int, mem: dict, *, resume: Optional[dict] = None, return_state: bool = False,
                planned_epochs: Optional[int] = None) -> dict:
    """LeJEPA+SIGReg global (mesma chamada em f_low, no treino contínuo e na retomada de f_high)."""
    from ga_search.modehb.external_seed import ElementProtect
    from ga_search.modehb.run_search import _amp_kw, _NullRun
    from ga_search.modehb.ssl_trainers import train_global_ssl
    ev, dev = gev.ev, gev.device
    tr = gev.cfg["train"]
    enc = (_CkptSequential if mem["grad_checkpoint"] else nn.Sequential)(*model.blocks)
    if mem["ssl_amp_bf16"]:                   # OPÇÃO DESLIGADA: muda a ciência (declarada no resultado)
        amp_kw = {"amp_dtype": torch.bfloat16}
    else:
        amp_kw = _amp_kw(ev)
    lr = float(gcfg.get("lr", tr.get("lr_new", 1e-3)))
    extra = {"resume": resume, "return_state": True} if (resume is not None or return_state) else {}
    v3 = _v3(gcfg)                            # growth v3 (2)(3): opt-in; padrões = chamada histórica
    if v3.get("aug") not in (None, "default"):
        extra["aug_cfg"] = v3["aug"]
    if v3.get("freeze_head_in_ssl"):
        extra["freeze_head"] = True
    if v3.get("vicreg_kwargs"):
        extra["vicreg_kwargs"] = dict(v3["vicreg_kwargs"])
    rec = ssl_recipe(gcfg)                    # receita do artigo (fiel/adaptação/pendente registrados)
    lam = float(tr.get("sigreg_lambda", 0.05))
    loader = ev.train_loader
    batch_info = None
    if rec is not None:
        kw = dict(rec["kwargs"])
        lam = float(kw.pop("sigreg_lambda", lam))
        if gcfg.get("ssl") != "lejepa":
            kw.pop("projector", None)
        extra.update(kw)
        if v3.get("aug") in (None, "default"):
            extra["aug_cfg"] = "strong"
        sb = v3.get("ssl_batch_size")         # v3.1 (bloco 2): batch do SSL da receita (maior que couber)
        if sb:
            B = int((resume or {}).get("ssl_batch") or sb)
            loader = _ssl_loader(ev, B, seed)
            spe = len(loader) if tr.get("max_steps_per_epoch") is None else min(len(loader), int(tr["max_steps_per_epoch"]))
            om = recipe_optim(rec, loader.batch_size, spe, int(planned_epochs or epochs))
            if om:
                lr = om["lr"]
                extra.update(optimizer=om["optimizer"], lr_schedule=om["lr_schedule"], wd_exclude_1d=om["wd_exclude_1d"])
                if "dino_kwargs" in extra:        # DINO: tau_t/freeze por época real, momentum pelos passos da agenda
                    extra["dino_kwargs"] = {**extra["dino_kwargs"], "steps_per_epoch": int(spe),
                                            "total_steps": om["lr_schedule"]["total_steps"]}
                if "byol_kwargs" in extra:        # EMA tau_k sobre os mesmos passos planejados da agenda
                    extra["byol_kwargs"] = {**extra["byol_kwargs"], "total_steps": om["lr_schedule"]["total_steps"]}
            batch_info = {"requested": int(sb), "used": int(loader.batch_size), "steps_per_epoch": int(spe),
                          "epoch": "passada completa do treino (sem max_images_per_epoch)", "optim": om}
    elif v3.get("weight_decay") is not None:  # correção do wd=0 efetivo (run_search não repassa): só na v3
        extra["weight_decay"] = float(v3["weight_decay"])
    head_for_ssl, gh, old_rg = model.head, None, None
    if v3.get("cov_gate_head"):               # growth v3.1: cabeça gated (antiga congelada + g_h * Cov)
        old_rg = [(p, p.requires_grad) for p in model.head.parameters()]
        for p, _ in old_rg:
            p.requires_grad_(False)
        cin_h = _block_out(model.spec, len(model.blocks) - 1)
        gh = CovGatedHead(model.head, cin_h, int(v3.get("cov_gate_head_k", 16)), seed=int(seed))
        if resume is not None and (resume.get("extra") or {}).get("_cov_gate_head") is not None:
            gh.load_state_dict(resume["extra"]["_cov_gate_head"])
        head_for_ssl = gh
        extra.pop("freeze_head", None)        # a cabeça antiga já está congelada; só g_h treina
    snap0 = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()} if batch_info else None
    gh0 = {k: v.detach().clone() for k, v in gh.state_dict().items()} if (batch_info and gh is not None) else None
    while True:
        try:
            with ElementProtect(enc, _masks_under(masks, "blocks."), scale), \
                    ElementProtect(model.head, _masks_under(masks, "head."), scale):
                ssl_log = train_global_ssl(enc, head_for_ssl, loader, gcfg.get("ssl", "lejepa"), gcfg.get("reg", "sigreg"),
                                           epochs, lr, lr, [], dev, _NullRun(), int(seed), lam,
                                           max_steps_per_epoch=tr.get("max_steps_per_epoch"), conv_lr_scale=1.0,
                                           progress=f"[ssl] {epochs} ép.", **{"projector": False, **amp_kw, **extra})
            break
        except Exception as exc:              # v3.1: o maior batch que couber (sem acumulação de gradiente)
            B = int(loader.batch_size) if batch_info else 0
            if not (batch_info and _is_oom(exc) and resume is None and B // 2 >= int(v3.get("ssl_batch_min", 32))):
                raise
            print(f"[ssl] OOM com batch {B}: refaz com {B // 2} (adaptação registrada)", flush=True)
            model.load_state_dict(snap0)
            if gh0 is not None:
                gh.load_state_dict(gh0)
            for p_ in model.parameters():
                p_.grad = None
            _free(dev, True)
            loader = _ssl_loader(ev, B // 2, seed)
            spe = len(loader) if tr.get("max_steps_per_epoch") is None else min(len(loader), int(tr["max_steps_per_epoch"]))
            om = recipe_optim(rec, loader.batch_size, spe, int(planned_epochs or epochs))
            if om:
                lr = om["lr"]
                extra.update(optimizer=om["optimizer"], lr_schedule=om["lr_schedule"], wd_exclude_1d=om["wd_exclude_1d"])
            batch_info.update(used=int(loader.batch_size), steps_per_epoch=int(spe), optim=om,
                              oom_halvings=int(batch_info.get("oom_halvings", 0)) + 1)
    del snap0
    if gh is not None:
        for p, rg in old_rg:
            p.requires_grad_(rg)
        ssl_log["cov_gate_head"] = {"g_h": float(gh.g.detach().cpu()), "k": int(gh.P.shape[1]),
                                    "dims_extra": int(gh.iu.shape[1])}
        if isinstance(ssl_log.get("state"), dict):
            ssl_log["state"].setdefault("extra", {})["_cov_gate_head"] = {k: v.detach().cpu().clone()
                                                                          for k, v in gh.state_dict().items()}
    if rec is not None:
        ssl_log["recipe"] = {k: rec[k] for k in ("fiel", "adaptacao", "pendente")}
        if batch_info is not None:            # v3.1: o que foi implementado sai de "pendente"
            ssl_log["recipe"]["pendente"] = [x for x in rec["pendente"] if "LARS" not in x and "warmup" not in x]
            ssl_log["recipe"]["batch"] = batch_info
            if isinstance(ssl_log.get("state"), dict):
                ssl_log["state"]["ssl_batch"] = int(batch_info["used"])
        ssl_log["recipe"]["kwargs"] = {k: v for k, v in rec["kwargs"].items()}
        bsz = int(batch_info["used"]) if batch_info is not None else int(gev.cfg["data"].get("batch_size", 64))
        ssl_log["recipe"]["batch_size"] = bsz
        ssl_log["recipe"]["variant"] = rec.get("variant")
        if batch_info is not None and batch_info["used"] < batch_info["requested"]:
            ssl_log["recipe"]["adaptacao"] = list(ssl_log["recipe"]["adaptacao"]) + [
                f"adaptação: batch {bsz} por memória (pedido {batch_info['requested']})"]
        if bsz < 128:
            ssl_log["recipe"]["adaptacao"] = list(ssl_log["recipe"]["adaptacao"]) + [
                f"batch {bsz} (< 128 do artigo; rank da cov <= N-1)"]
    del enc
    return ssl_log


def _wise_select(gev: GrowthEvaluator, model: BB.Backbone, anchor: dict, trained: dict, ssl_finite: bool,
                 gcfg: dict, mem: dict):
    """WiSE: alpha pela ORIG-val entre âncora (morfismo = pai) e treinado; o modelo termina no melhor blend."""
    dev = gev.device
    alphas = [float(a) for a in gcfg.get("wise_alphas", [0.0, 0.25, 0.5, 0.75, 1.0])]
    if 0.0 not in alphas:
        alphas = [0.0] + alphas
    if not ssl_finite:                     # SSL não finito: só alpha=0 (= pai) é avaliado
        alphas = [0.0]
    gev.reset_stamps()
    # growth_fix7: leituras (GPU) em sequência, depois as SVMs em paralelo (score_many); mesma ordem e mesma
    # regra de desempate do laço sequencial -> mesmo alpha, mesmos kappas.
    feats, heads = [], []
    for a in alphas:
        _blend_into(model, anchor, trained, a)
        gev._last_head = None
        feats.append(gev.features(model, embed_batch_size=mem["embed_batch_size"]))
        heads.append(gev._last_head)              # growth v3: cabeça so2 ajustada para este alpha (None = head)
        _free(dev, mem["empty_cache"])
    svs = gev.score_many(feats)
    del feats
    best, kappas, bi = None, {}, 0
    for i, (a, sv) in enumerate(zip(alphas, svs)):
        kappas[str(a)] = float(sv["kappa"])
        if best is None or sv["quality"] > best[1]["quality"]:
            best, bi = (a, sv), i
    _blend_into(model, anchor, trained, best[0])
    model.eval()
    model.__dict__["so2_head"] = heads[bi]        # fora do state_dict (sidecar so2_head.pt)
    return best, kappas


def train_child(gev: GrowthEvaluator, res: MO.GrowthResult, fidelity: int, gcfg: dict, seed: int,
                parent_kappa: float, *, epochs: Optional[Tuple[int, int]] = None,
                capture: Optional[dict] = None, anchor_state: Optional[dict] = None) -> Tuple[BB.Backbone, dict]:
    """AE local (blocos novos) -> LeJEPA+SIGReg -> WiSE (âncora = filho no passo 0 = pai) -> kappa ORIG-val.

    growth_fix6: ``epochs=(ae, ssl)`` força a divisão (padrão: ``split_epochs(fidelity)``; o treino contínuo de
    referência da retomada usa (ae(f_low), f_high - ae(f_low))). ``capture`` (dict) recebe o estado de retomada:
    pesos treinados PRÉ-WiSE, âncora, máscaras e o estado do SSL (AdamW/RNG/passo)."""
    from ga_search.modehb.external_seed import ElementProtect
    from ga_search.modehb.local_ae import train_local_ae
    from ga_search.modehb.run_search import _amp_kw, _NullRun, split_epochs
    from ga_search.modehb import modehb as M
    ev, dev = gev.ev, gev.device
    tr = gev.cfg["train"]
    mem = _mem(gcfg)
    model = res.child.to(dev)
    masks = {k: v.to(dev) for k, v in res.masks.items()}
    scale = PROTECT_SCALE[gcfg["protect"]]
    snap = _snapper(mem)
    anchor = snap(model)
    if anchor_state is not None:             # growth v3 (4f): op inexata -> âncora do WiSE = o PAI (função), alpha=0 = pai
        anchor = {k: (v.detach().to("cpu", copy=True) if mem["wise_on_cpu"] else v.detach().to(dev, copy=True))
                  for k, v in anchor_state.items()}
    ae_ep, ssl_ep = split_epochs(fidelity, tr.get("ae_ssl_split", [1, 2])) if epochs is None \
        else (int(epochs[0]), int(epochs[1]))
    run = _NullRun()
    new_blocks = [j for j in range(len(model.blocks))
                  if any(not bool(masks[f"blocks.{j}.{k}"].all()) for k, _ in model.blocks[j].named_parameters())]
    ae_logs = []
    ae_loader = ev.train_loader
    if _v3(gcfg).get("full_epoch_ae"):       # v3.1 bloco 6: época do AE = passada completa (sem max_images_per_epoch)
        ae_loader = _ssl_loader(ev, int(gev.cfg["data"].get("batch_size", 64)), int(seed) + 17)
    torch.manual_seed(int(seed))
    for j in new_blocks:
        for p in model.parameters():           # o prefixo não pode carregar gradiente do AE anterior
            p.grad = None
        prefix = nn.Sequential(*model.blocks[:j]).to(dev).eval()
        prefix.out_channels = _block_in(model.spec, j)
        blk = model.blocks[j]
        # growth_mem2: checkpoint do bloco novo no AE local (mesmos parâmetros/ordem; BN/DBN 1x; bit a bit)
        # v3.1 (velocidade): checkpoint do bloco no AE local é bit a bit igual mas custa ~30-40% do passo (perfil
        # CPU, 224 px: recomputa conv+DBN+act+pool e o _ActPool fatiado) e quase não poupa memória num bloco só
        # (growth_mem2). ``mem.ae_grad_checkpoint`` (ausente = segue grad_checkpoint, histórico).
        ae_ck = mem.get("ae_grad_checkpoint")
        ae_ck = mem["grad_checkpoint"] if ae_ck is None else bool(ae_ck)
        ae_blk = _CkptSequential(blk) if ae_ck else blk
        akw = ae_extra_kw(gcfg)                # hotfix ae_diverge (ausente = chamada histórica)
        pre_ae = {k: v.detach().clone() for k, v in blk.state_dict().items()} \
            if akw.get("diverge_ratio") is not None else None
        with ElementProtect(blk, _masks_under(masks, f"blocks.{j}."), scale):
            ae = train_local_ae(prefix, ae_blk, ae_loader, gcfg.get("ae_target", "pixel"), ae_ep,
                                float(gcfg.get("lr_ae", tr.get("lr_ae", 1e-3))), dev, run, int(seed) + j,
                                conv_lr_scale=1.0, decoder_warmup_frac=M.AE_WARMUP_FRAC[gcfg.get("ae_schedule", "joint")],
                                progress=f"[ae] bloco {j}", **_amp_kw(ev), **akw)
        rec_ae = {"block": j, **{k: v for k, v in ae.items() if isinstance(v, (int, float, str, bool))}}
        if pre_ae is not None:
            rec_ae["ae_diverged"] = bool(ae.get("diverged"))
            if ae.get("diverged"):               # guarda: o bloco volta ao morfismo (pré-AE); o filho segue sem AE nele
                with torch.no_grad():
                    blk.load_state_dict(pre_ae)
                rec_ae["ae_restored"] = True
                print(f"[ae] bloco {j}: ae_diverged ({ae.get('diverge_reason')}) -> bloco restaurado ao "
                      f"estado pré-AE (morfismo)", flush=True)
            del pre_ae
        rr = rec_ae.get("recon_ratio")
        if rr is not None and (not math.isfinite(float(rr)) or float(rr) > 1.0):
            rec_ae["ae_alert"] = f"recon_ratio={float(rr):.4g} > 1"
            print(f"[ae] ALERTA (hotfix ae_diverge): bloco {j} recon_ratio={float(rr):.4g} > 1 "
                  f"(perda {rec_ae.get('initial_recon_loss')} -> {rec_ae.get('final_recon_loss')}; "
                  f"restaurado={bool(rec_ae.get('ae_restored'))})", flush=True)
        ae_logs.append(rec_ae)
        del prefix, ae
        _free(dev, mem["empty_cache"])
    ae_alerts = [r for r in ae_logs if r.get("ae_alert")]
    if ae_alerts and ae_smoke_strict(gcfg):   # hotfix ae_diverge: o smoke falha com recon_ratio > 1
        raise ChainError("AE local com recon_ratio > 1 (smoke estrito): " + "; ".join(
            f"bloco {r['block']} {r['ae_alert']} restaurado={bool(r.get('ae_restored'))}" for r in ae_alerts))
    for p in model.parameters():
        p.grad = None
        p.requires_grad_(True)
    ssl_off = ssl_disabled(gcfg)
    if ssl_off:                              # growth_fix4: ablação ``ssl: none`` (só o AE local nos blocos novos)
        ssl_log = {"steps": 0, "final_loss_total": 0.0, "skipped": True}
    else:
        planned = ssl_ep
        fl_, fh_ = int(gcfg.get("fid_low", 0) or 0), int(gcfg.get("fid_high", 0) or 0)
        if capture is not None and fh_ > fl_ and int(fidelity) == fl_:      # v3.1: agenda até a fidelidade alta
            planned = int(ssl_ep) + resume_extra_epochs(fl_, fh_, gcfg, tr)
        ssl_log = _global_ssl(gev, model, masks, scale, ssl_ep, gcfg, seed, mem, return_state=capture is not None,
                              planned_epochs=planned)
    for p in model.parameters():
        p.grad = None
    _free(dev, mem["empty_cache"])
    zca_log = _zca_refit(gev, model, new_blocks, gcfg) if _v3(gcfg).get("zca_refit") else None
    trained = snap(model)
    ssl_finite = _finite_state(trained) and math.isfinite(float(ssl_log.get("final_loss_total") or 0.0))
    if capture is not None:                  # growth_fix6: estado de retomada (pesos PRÉ-WiSE)
        capture.update(trained={k: v.detach().to("cpu", copy=True) for k, v in trained.items()},
                       anchor={k: v.detach().to("cpu", copy=True) for k, v in anchor.items()},
                       masks={k: v.detach().to("cpu", copy=True) for k, v in masks.items()},
                       ssl_state=ssl_log.pop("state", None), ae_epochs=int(ae_ep), ssl_epochs=int(ssl_ep),
                       ssl_finite=bool(ssl_finite), ssl_skipped=bool(ssl_off), anchor_is_parent=anchor_state is not None)
    best, kappas = _wise_select(gev, model, anchor, trained, ssl_finite, gcfg, mem)
    k = float(best[1]["kappa"])
    out = {"kappa_val_orig": k, "quality": FV.quality_of(k), "wise_alpha": float(best[0]),
           "aux": {"wise_kappas": kappas, "kappa_val_alpha0": kappas.get("0.0"),
                   "macro_f1": float(best[1].get("macro_f1", float("nan")))},
           "decision_rulers": gev.decision_rulers(), "n_svm_decisions": len(ev._svm_stamps),
           "_val_pred": best[1].get("pred"), "_val_true": best[1].get("y_true"),
           "ae_epochs": ae_ep, "ssl_epochs": ssl_ep, "ae_blocks": new_blocks, "ae_logs": ae_logs,
           "ae_diverged_blocks": [r["block"] for r in ae_logs if r.get("ae_diverged")],
           "ae_alerts": [{"block": r["block"], "alert": r["ae_alert"]} for r in ae_alerts],
           "ae_full_epoch": bool(_v3(gcfg).get("full_epoch_ae")),
           "ssl_final_loss": ssl_log.get("final_loss_total"), "ssl_steps": int(ssl_log.get("steps", 0)),
           "ssl_finite": bool(ssl_finite), "parent_kappa_val_orig": float(parent_kappa),
           "ssl": "none" if ssl_off else gcfg.get("ssl", "lejepa"), "ssl_skipped": bool(ssl_off),
           "augment_device": ssl_log.get("augment_device", str(dev)),   # growth_fix8: vistas no device do treino
           "embed_precision": gev.embed_precision,
           "mem": dict(mem),
           "status": "ok"}
    _v3_fields(gev, model, out, ssl_log, gcfg, anchor_is_parent=anchor_state is not None)
    if zca_log is not None:
        out["zca_refit"] = zca_log
    tg = _trained_gates(trained)
    if tg:
        out["cov_gates_trained"] = tg            # growth v3.1: g aprendido no AE/SSL (antes do WiSE)
    FV.check_decisions(out, gev.fitness_spec)     # toda decisão (alpha e kappa) pela régua ORIG-val
    return model, out


@torch.no_grad()
def _zca_refit(gev: GrowthEvaluator, model: BB.Backbone, blocks: Sequence[int], gcfg: dict) -> List[dict]:
    """v3.1 (whitening.md): ZCA estatística dos blocos NOVOS re-ajustada nas respostas de TREINO depois do AE/SSL
    (a conv mudou; o branqueamento congelado do morfismo ficou velho). O WiSE mistura também mu/W: alpha=0
    devolve o branqueamento do pai/morfismo bit a bit."""
    from ga_search.modehb.ops_norm_act import ZCAWhitening
    n = int(_v3(gcfg).get("zca_refit_images", 256))
    x = gev.ev.svm_train_x[:n]
    dev = gev.device
    model.eval()
    log = []
    for j in sorted(set(int(b) for b in blocks)):
        blk = model.blocks[j]
        zs = [(nm, m) for nm, m in blk.named_modules() if isinstance(m, ZCAWhitening)]
        if not zs or not hasattr(blk, "conv"):
            continue
        feats = []
        for i0 in range(0, len(x), 32):
            h = x[i0:i0 + 32].to(dev).float()
            for b in model.blocks[:j]:
                h = b(h)
            feats.append(blk.conv(h).float())
        f = torch.cat(feats)
        for nm, z in zs:
            before = z.W.detach().clone()
            z.fit(f)
            log.append({"block": j, "module": nm or "norm", "n_images": int(x.shape[0]), "split": "train",
                        "W_rel_change": float((z.W - before).norm() / before.norm().clamp_min(1e-12))})
    return log


def _trained_gates(sd: dict) -> dict:
    """g de cada bloco cov_gate num state_dict (pré-WiSE): o que o AE/SSL aprendeu."""
    out = {}
    for k, v in sd.items():
        parts = k.split(".")
        if len(parts) == 3 and parts[0] == "blocks" and parts[2] == "gate":
            g = v.detach().float().cpu()
            out[parts[1]] = {"mean": float(g.mean()), "mean_abs": float(g.abs().mean()), "max_abs": float(g.abs().max())}
    return out


def _v3_fields(gev: GrowthEvaluator, model, out: dict, ssl_log: dict, gcfg: dict, anchor_is_parent: bool = False):
    """growth v3: registro de readout/cabeça so2, SSL/reg efetivos, DA, cabeça congelada e aux GAP (nunca decide)."""
    v3 = _v3(gcfg)
    out["readout"] = getattr(gev, "readout", "head")
    out["ssl_method"] = ssl_log.get("method", out.get("ssl"))
    out["reg"] = None if out.get("ssl_skipped") else gcfg.get("reg", "sigreg")
    out["aug"] = ssl_log.get("aug") or ({"preset": "historical"} if not out.get("ssl_skipped") else None)
    out["aug_gene"] = v3.get("aug")
    out["freeze_head_in_ssl"] = bool(ssl_log.get("freeze_head", False))
    out["whiten_new"] = v3.get("whiten_new")
    out["softhebb"] = bool(v3.get("softhebb"))
    out["anchor"] = "parent_function (pré-op inexata)" if anchor_is_parent else "morphism_step0 (= pai em função)"
    if ssl_log.get("recipe"):
        out["ssl_recipe"] = ssl_log["recipe"]
    cg = BB.cov_gate_values(model)            # growth v3.1: valores finais de g por bloco (diagnóstico)
    if cg:
        out["cov_gates"] = cg
    if ssl_log.get("cov_gate_head"):
        out["cov_gate_head"] = ssl_log["cov_gate_head"]
    out["spifil_images_per_class"] = gcfg.get("spifil_images_per_class")
    out["spifil_n_superpixels"] = v3.get("spifil_n_superpixels")
    if getattr(model, "so2_head", None) is not None:
        out["so2_head_meta"] = model.so2_head.meta()
        out["so2_head_sha256"] = model.so2_head.sha256()
    ag = gev.aux_gap(model) if hasattr(gev, "aux_gap") else None
    if ag is not None:
        out["aux"]["gap"] = ag


# ═════════════════════════════════════════════════════════════════ 5b. growth_fix6: retomada f_low -> f_high
RESUME_FORMAT = "growth_resume/v1"
RESUME_FILE = "resume_state.pt"


def resume_extra_epochs(fid_low: int, fid_high: int, gcfg: dict, train_cfg: Optional[dict] = None) -> int:
    """Épocas de SSL que a retomada acrescenta: fid_high - fid_low ("fid_diff", padrão: o nó termina com
    ae(f_low) + ssl(f_low) + (f_high - f_low) = f_high épocas) ou ssl(f_high) - ssl(f_low) ("ssl_diff")."""
    from ga_search.modehb.run_search import split_epochs
    if _promo(gcfg)["extra_epochs"] == "ssl_diff":
        sp = (train_cfg or {}).get("ae_ssl_split", [1, 2])
        return max(0, split_epochs(fid_high, sp)[1] - split_epochs(fid_low, sp)[1])
    return max(0, int(fid_high) - int(fid_low))


def save_resume_state(path, capture: dict, *, save_optimizer: bool, meta: dict) -> dict:
    """Grava (atômico) o estado de retomada de f_low: pesos treinados PRÉ-WiSE, âncora (morfismo), máscaras de
    proteção e o estado do SSL (AdamW só se ``save_optimizer``; RNG e passo sempre). -> {path, sha256, ...}."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ssl_state = dict(capture.get("ssl_state") or {}) or None
    has_opt = bool(ssl_state is not None and ssl_state.get("opt") is not None and save_optimizer)
    if ssl_state is not None and not save_optimizer:
        ssl_state["opt"] = None
    payload = {"format": RESUME_FORMAT, "trained": capture["trained"], "anchor": capture["anchor"],
               "masks": capture["masks"], "ssl_state": ssl_state, "ae_epochs": int(capture["ae_epochs"]),
               "ssl_epochs": int(capture["ssl_epochs"]), "ssl_finite": bool(capture.get("ssl_finite", True)),
               "anchor_is_parent": bool(capture.get("anchor_is_parent", False)), "meta": _jsonable(meta)}
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with tmp.open("wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return {"path": str(path), "sha256": I.file_sha256(path), "has_optimizer": has_opt,
            "has_rng": bool(ssl_state is not None and ssl_state.get("rng") is not None),
            "format": RESUME_FORMAT}


def load_resume_state(path, expected_sha256: Optional[str] = None) -> dict:
    sha = I.file_sha256(path)
    if sha is None:
        raise FileNotFoundError(str(path))
    if expected_sha256 and sha != expected_sha256:
        raise ChainError(f"{path}: sha256 {sha} != {expected_sha256}")
    st = torch.load(path, map_location="cpu", weights_only=False)
    if st.get("format") != RESUME_FORMAT:
        raise ChainError(f"{path}: formato {st.get('format')!r}")
    return st


def continue_child(gev: GrowthEvaluator, model: BB.Backbone, state: dict, extra_epochs: int, gcfg: dict,
                   seed: int, parent_kappa: float, *, capture: Optional[dict] = None) -> Tuple[BB.Backbone, dict]:
    """growth_fix6: RETOMA o filho de f_low (``model`` já com os pesos treinados PRÉ-WiSE de ``state``) e treina
    ``extra_epochs`` épocas de SSL com o mesmo lr (fixo), o AdamW e as RNGs salvos (sem AdamW salvo: AdamW novo,
    registrado). O AE local NÃO é repetido. WiSE entre a âncora de f_low (o morfismo) e o treinado-f_high."""
    dev = gev.device
    mem = _mem(gcfg)
    model = model.to(dev)
    masks = {k: v.to(dev) for k, v in state["masks"].items()}
    scale = PROTECT_SCALE[gcfg["protect"]]
    anchor = state["anchor"] if mem["wise_on_cpu"] else {k: v.to(dev) for k, v in state["anchor"].items()}
    ssl_state = state.get("ssl_state")
    opt_resumed = bool(ssl_state is not None and ssl_state.get("opt") is not None)
    rng_resumed = bool(ssl_state is not None and ssl_state.get("rng") is not None)
    for p in model.parameters():
        p.grad = None
        p.requires_grad_(True)
    resume = dict(ssl_state) if ssl_state is not None else {"opt": None, "rng": None, "extra": {}, "step": 0}
    ssl_log = _global_ssl(gev, model, masks, scale, int(extra_epochs), gcfg, seed, mem, resume=resume,
                          return_state=True, planned_epochs=int(state["ssl_epochs"]) + int(extra_epochs))
    for p in model.parameters():
        p.grad = None
    _free(dev, mem["empty_cache"])
    trained = _snapper(mem)(model)
    ssl_finite = _finite_state(trained) and math.isfinite(float(ssl_log.get("final_loss_total") or 0.0))
    if capture is not None:
        capture.update(trained={k: v.detach().to("cpu", copy=True) for k, v in trained.items()},
                       anchor=state["anchor"], masks=state["masks"], ssl_state=ssl_log.pop("state", None),
                       ae_epochs=int(state["ae_epochs"]), ssl_epochs=int(state["ssl_epochs"]) + int(extra_epochs),
                       ssl_finite=bool(ssl_finite), ssl_skipped=False)
    best, kappas = _wise_select(gev, model, anchor, trained, ssl_finite, gcfg, mem)
    k = float(best[1]["kappa"])
    out = {"kappa_val_orig": k, "quality": FV.quality_of(k), "wise_alpha": float(best[0]),
           "aux": {"wise_kappas": kappas, "kappa_val_alpha0": kappas.get("0.0"),
                   "macro_f1": float(best[1].get("macro_f1", float("nan")))},
           "decision_rulers": gev.decision_rulers(), "n_svm_decisions": len(gev.ev._svm_stamps),
           "ae_epochs": int(state["ae_epochs"]), "ssl_epochs": int(state["ssl_epochs"]) + int(extra_epochs),
           "ae_blocks": [], "ae_logs": [], "ssl_final_loss": ssl_log.get("final_loss_total"),
           "ssl_steps": int(ssl_log.get("steps", 0)), "ssl_finite": bool(ssl_finite),
           "parent_kappa_val_orig": float(parent_kappa), "ssl": gcfg.get("ssl", "lejepa"), "ssl_skipped": False,
           "augment_device": ssl_log.get("augment_device", str(gev.device)),   # growth_fix8
           "embed_precision": gev.embed_precision, "mem": dict(mem),
           "resume": {"extra_ssl_epochs": int(extra_epochs), "optimizer": "resumed" if opt_resumed else
                      "restarted (AdamW não salvo em f_low)", "optimizer_resumed": opt_resumed,
                      "rng_resumed": rng_resumed, "ae_repeated": False},
           "status": "ok"}
    _v3_fields(gev, model, out, ssl_log, gcfg, anchor_is_parent=bool(state.get("anchor_is_parent", False)))
    FV.check_decisions(out, gev.fitness_spec)
    return model, out


_SEL_FROM_LOW = ("kappa_val_orig", "quality", "wise_alpha", "checkpoint", "checkpoint_sha256", "aux", "so2_head",
                  "so2_head_sha256", "so2_head_meta", "band", "params", "flops_2mac", "flops", "arch", "delta_kappa",
                  "kappa_parent_rescored", "parent_kappa_val_orig", "decision_rulers", "preservation",
                  "preservation_rel_l2_emb", "preservation_verdict", "bootstrap", "cov_gates", "cov_gates_trained",
                  "cov_gate_head", "ssl_recipe", "growth_spec_id", "status")


def select_low_high(r_low: dict, r_high: dict, fl: int, fh: int) -> dict:
    """HOTFIX member_fid: o resultado PROMOVIDO é o melhor de f_low/f_high pela ORIG-val (empate = f_high; f_high
    falho = f_low). Vale em todos os caminhos (retomada, retrain, ssl: none). Quando f_low vence, checkpoint, kappa,
    alpha, aux (wise_kappas), so2 e o resto da decisão vêm de f_low; a cand_key continua a da promoção (linhagem)."""
    r = dict(r_high)
    if r.get("selected_fid") is not None and int(r["selected_fid"]) == int(fl):
        return r                                    # a retomada já escolheu f_low (fix6)
    k_low = r_low.get("kappa_val_orig")
    if k_low is None or r_low.get("status", "ok") != "ok":   # f_low sem kappa (não deveria ser promovido): fica f_high
        r.setdefault("selected_fid", int(fh))
        return r
    k_high = r.get("kappa_val_orig") if r.get("status") == "ok" else None
    pick_high = k_high is not None and best_of_low_high(float(k_low), float(k_high)) == 1
    r.setdefault("kappa_val_orig_f_low", float(k_low))
    r.setdefault("wise_alpha_f_low", r_low.get("wise_alpha"))
    r.setdefault("kappa_val_orig_f_high", None if k_high is None else float(k_high))
    r.setdefault("wise_alpha_f_high", r.get("wise_alpha"))
    r["fid_low"], r["fid_high"] = int(fl), int(fh)
    if pick_high:
        r["selected_fid"] = int(fh)
        return r
    r["checkpoint_f_high"], r["checkpoint_sha256_f_high"] = r_high.get("checkpoint"), r_high.get("checkpoint_sha256")
    r["status_f_high"] = r_high.get("status")
    r["aux_f_high"] = {"wise_kappas": (r_high.get("aux") or {}).get("wise_kappas")}
    for k in _SEL_FROM_LOW:
        if k in r_low:
            r[k] = r_low[k]
    r["selected_fid"] = int(fl)
    r["selection"] = ("best_of_low_high (hotfix member_fid): f_low venceu pela ORIG-val"
                      + ("" if k_high is not None else " (f_high falhou)") + "; teste nunca lido")
    return r


def best_of_low_high(k_low: float, k_high: float) -> int:
    """Seleção do nó promovido pela ORIG-val: f_high, a não ser que kappa_val(f_high) < kappa_val(f_low)
    (empate fica com f_high). Devolve 0 (f_low) ou 1 (f_high). Nunca usa teste."""
    if k_high is None or not math.isfinite(float(k_high)):
        return 0
    return 0 if float(k_high) < float(k_low) else 1


# ═════════════════════════════════════════════════════════════════ 6. utilidades de seleção
def _kappa_np(y: np.ndarray, p: np.ndarray, k: int) -> float:
    """Cohen kappa (não ponderado), = multiclass_cohen_kappa do core.metrics."""
    cm = np.zeros((k, k), dtype=np.float64)
    np.add.at(cm, (y, p), 1.0)
    n = cm.sum()
    po = np.trace(cm) / n
    pe = float((cm.sum(0) * cm.sum(1)).sum()) / (n * n)
    return 0.0 if pe >= 1.0 else float((po - pe) / (1.0 - pe))


def bootstrap_delta_kappa(parent_pred: np.ndarray, child_pred: np.ndarray, y: np.ndarray, *, n: int = 1000,
                          alpha: float = 0.05, seed: int = 0) -> dict:
    """v3.1: bootstrap PAREADO da validação (mesmas amostras para pai e filho): Δκ = κ_filho - κ_pai.
    ``accept_ok`` = quantil alpha de Δκ >= 0 (melhora significativa). Só validação; o teste nunca entra."""
    k = int(max(y.max(), parent_pred.max(), child_pred.max())) + 1
    rng = np.random.default_rng(int(seed))
    m = len(y)
    d = np.empty(int(n))
    for b in range(int(n)):
        i = rng.integers(0, m, m)
        d[b] = _kappa_np(y[i], child_pred[i], k) - _kappa_np(y[i], parent_pred[i], k)
    lo, hi = float(np.quantile(d, alpha)), float(np.quantile(d, 1 - alpha))
    return {"delta_kappa_point": _kappa_np(y, child_pred, k) - _kappa_np(y, parent_pred, k),
            "delta_kappa_ci": [lo, hi], "alpha": float(alpha), "n": int(n), "p_improve": float((d > 0).mean()),
            "accept_ok": bool(lo >= 0.0), "split": "validation"}


def decide(r: dict, delta: float, require_improvement: bool) -> Tuple[str, bool]:
    """growth_fix4 (b). -> (decisão, entra na faixa?). Preservação exata + WiSE alpha=0 => o filho É o pai:
    ``adopted_as_parent`` (nem aceito nem rejeitado por ruído; não ocupa a faixa, não é promovido). Senão,
    aceitação real: kappa_filho >= kappa_pai_re-pontuado (mesma leitura, mesmo processo) + delta."""
    if r.get("status") != "ok":
        return "failed", False
    pk0 = r.get("kappa_parent_rescored")
    same = pk0 is None or r.get("kappa_val_orig") is None or \
        math.isclose(float(r["kappa_val_orig"]), float(pk0), abs_tol=1e-12)
    # growth v3: com a cabeça so2, alpha=0 + preservação exata do EMBEDDING antigo não implica a mesma leitura
    # (canais novos no último bloco entram na so2): só é "o pai" se o kappa também é o do pai; senão vale a regra.
    if str(r.get("readout", "head")) == "head":
        same = True                              # histórico (fix4): alpha=0 + exato = o pai, mesmo com ruído no kappa
    if float(r.get("wise_alpha", 0.0)) == 0.0 and r.get("preservation_verdict") == "exact" and same:
        return "adopted_as_parent", False
    pk = r.get("kappa_parent_rescored")
    if pk is None:                              # resultado anterior ao fix4 (não deveria ser adotado: fitness_sig)
        pk = r.get("parent_kappa_val_orig")
    ok = accept(r.get("kappa_val_orig"), pk, delta, r.get("wise_alpha", 0.0), require_improvement)
    bt = r.get("bootstrap")
    if ok and isinstance(bt, dict) and bt.get("accept_ok") is False:   # v3.1: margem estatística (opção registrada)
        return "rejected_not_significant", False
    return ("accepted" if ok else "rejected"), ok


def accept(child_kappa: float, parent_kappa: float, delta: float, alpha: float, require_improvement: bool) -> bool:
    """Regra de aceitação: kappa_ORIG_val(filho) >= kappa_ORIG_val(pai) + delta (e alpha>0 se exigido)."""
    if child_kappa is None or not math.isfinite(float(child_kappa)):
        return False
    if require_improvement and float(alpha) == 0.0:
        return False
    return float(child_kappa) >= float(parent_kappa) + float(delta) - 1e-12


def pareto_order(items: List[dict]) -> List[dict]:
    """Frentes não dominadas (kappa máx, FLOPs 2·MAC mín); dentro da frente: kappa desc, FLOPs, params, id."""
    rest, out = list(items), []
    while rest:
        front = [a for a in rest if not any((b["kappa_val_orig"] >= a["kappa_val_orig"] and b["flops_2mac"] <= a["flops_2mac"])
                                            and (b["kappa_val_orig"] > a["kappa_val_orig"] or b["flops_2mac"] < a["flops_2mac"])
                                            for b in rest)]
        front.sort(key=lambda r: (-r["kappa_val_orig"], r["flops_2mac"], r["params"], r["node_id"]))
        out += front
        ids = {id(a) for a in front}
        rest = [a for a in rest if id(a) not in ids]
    return out


def de_theta(pool: List[List[float]], rng: np.random.Generator, F: float = 0.5, CR: float = 0.5) -> List[float]:
    """DE rand1/bin sobre os thetas aceitos da faixa (aleatório uniforme com < 3)."""
    if len(pool) < 3:
        return [float(x) for x in rng.uniform(0, 1, 4)]
    a, b, c = (np.asarray(pool[i]) for i in rng.choice(len(pool), 3, replace=False))
    v = np.clip(a + F * (b - c), 0, 1)
    tgt = np.asarray(pool[int(rng.integers(len(pool)))])
    m = rng.uniform(0, 1, 4) < CR
    m[int(rng.integers(4))] = True
    return [float(x) for x in np.where(m, v, tgt)]


def _h(obj) -> int:
    return int(hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:8], 16)


def _gpu_total_gib(device) -> Optional[float]:
    try:
        if torch.cuda.is_available() and getattr(device, "type", str(device)) == "cuda":
            idx = device.index if getattr(device, "index", None) is not None else torch.cuda.current_device()
            return torch.cuda.get_device_properties(idx).total_memory / _GIB
    except Exception:
        pass
    return None


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())) or "out of memory" in str(exc).lower()


# ═════════════════════════════════════════════════════════════════ 7. controlador
class GrowthChain:
    def __init__(self, cfg: dict, gcfg: dict, gev: GrowthEvaluator, out_dir, *, allow_drift: Sequence[str] = (),
                 python: str = sys.executable):
        self.cfg, self.g, self.gev = cfg, gcfg, gev
        self.out = Path(out_dir)
        self.allow_drift = list(allow_drift)
        self.python = python
        self.state_path = self.out / "growth_state.json"
        self.lineage_path = self.out / "lineage.jsonl"
        self.test_queue = self.out / "test_queue.jsonl"
        self.full_cfg = {"run": cfg, "growth": gcfg, "kind": KIND}

    # ---- estado
    def _fresh(self) -> dict:
        return {"kind": KIND, "fitness_sig": self.gev.fitness_sig, "fitness_spec": self.gev.fitness_spec,
                "config_full": json.loads(json.dumps(self.full_cfg, default=str)),
                "config_sha": I.config_sha(self.full_cfg), "drift_log": [],
                "members": {b: [] for b in BANDS}, "jobs": {}, "gens_done": [], "seeds_done": False,
                "created": time.time()}

    def _load(self) -> dict:
        st = I.read_json(self.state_path)
        if st is None:
            return self._fresh()
        if st.get("fitness_sig") != self.gev.fitness_sig:
            raise FV.FitnessSigMismatch(f"estado {st.get('fitness_sig')} != avaliador {self.gev.fitness_sig}")
        d = I.check_drift(st, self.full_cfg, self.allow_drift)
        if d is not None:
            st["drift_log"].append(d)
            st["config_full"] = json.loads(json.dumps(self.full_cfg, default=str))
            st["config_sha"] = I.config_sha(self.full_cfg)
        return st

    def _save(self):
        self.state["updated"] = time.time()
        I.atomic_write_json(self.state_path, self.state)

    def _lineage_keys(self) -> set:
        keys = set()
        if self.lineage_path.exists():
            for line in self.lineage_path.read_text().splitlines():
                try:
                    r = json.loads(line)
                    keys.add((r["cand_key"], r.get("retry_of")))
                except Exception:          # linha parcial de um kill -9: ignorada (a entrada é reescrita)
                    pass
        return keys

    def _lineage(self, rec: dict):
        # dedup por (cand_key, retry_of): um retry de OOM acrescenta registro novo; leitor usa o ÚLTIMO por cand_key
        k = (rec["cand_key"], rec.get("retry_of"))
        if k in self._lin_keys:
            return
        with self.lineage_path.open("a") as f:
            f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._lin_keys.add(k)

    # ---- sementes
    def _seed_sources(self) -> List[dict]:
        sc = self.g.get("seeds") or {}
        ds = self.gev.ev.dataset
        rows = []
        reg = sc.get("registry")
        if reg and Path(reg).exists():
            import csv
            for r in csv.DictReader(open(reg)):
                if r["dataset"] == ds and r["regua"] == sc.get("regua", "ORIG") and r["faixa"] in sc.get("bands", ["B0"]) \
                        and r.get("checkpoint") and r.get("utilizavel", "sim") == "sim":
                    rows.append({"kind": "production", "path": r["checkpoint"], "label": f"{r['busca']}:{r['cand']}:{r['fid']}",
                                 "registry_kappa_val": r.get("kappa_val"), "registry_md5": r.get("md5")})
        for p in sc.get("checkpoints") or []:
            rows.append({"kind": "production", "path": p, "label": Path(p).parent.parent.name})
        if sc.get("external_small"):
            rows.append({"kind": "external_small", "path": None, "label": "spifil_pequeno"})
        return rows

    def _seed_stage(self):
        size = int(self.cfg["data"].get("image_size", 224))
        x = self.gev.ev.val_x[:4].float()
        for i, src in enumerate(self._seed_sources()):
            node = f"seed{i:02d}"
            ndir = self.out / "nodes" / node
            ckey = I.cand_key(I.lineage_key({"seed": src}, int(self.cfg["seed"]), {"kind": src["kind"]},
                                            self.gev.fitness_sig), 0)
            done = I.read_json(ndir / "result.json")
            if done is not None and I.adoptable(done, ckey=ckey, fitness_sig=self.gev.fitness_sig) is None:
                rec = done
            else:
                try:
                    if src["kind"] == "production" and _is_backbone_ckpt(src["path"]):
                        model, _spec = BB_load(src["path"])
                        info = {"source": src["path"], "format": "backbone_ckpt/v1"}
                    elif src["kind"] == "production":
                        model, info = backbone_from_production(src["path"], self.gev.ev.dataset, size, verify_x=x)
                    else:
                        model, info = backbone_from_external_small(self.gev.ev.dataset, size,
                                                                   seed=int(self.cfg["seed"]))
                except (ChainError, FileNotFoundError, NotImplementedError, KeyError) as exc:
                    print(f"[growth] semente {src['label']} ignorada: {exc}", flush=True)
                    continue
                if model.spec.cost.params_total > HARD_CAP:
                    continue
                self.gev.reset_stamps()
                self.gev._last_head = None
                sv = self.gev.score(model)
                model.__dict__["so2_head"] = self.gev._last_head
                ck = BB_save(model, ndir / "backbone.pt")
                rec = {"status": "ok", "node_id": node, "band": band_of(model.spec.cost.params_total),
                       "kappa_val_orig": float(sv["kappa"]), "quality": FV.quality_of(sv["kappa"]),
                       "embed_precision": sv.get("embed_precision"),
                       "decision_rulers": self.gev.decision_rulers(), "params": int(model.spec.cost.params_total),
                       "flops_2mac": int(model.spec.cost.flops), "flops": int(model.spec.cost.macs),
                       "checkpoint": ck["path"], "checkpoint_sha256": ck["file_sha256"], "arch": model.spec.id,
                       "source": src, "info": _jsonable(info), "cand_key": ckey, "fitness_sig": self.gev.fitness_sig,
                       "fid": 0, "parent": None, "theta": None, "readout": getattr(self.gev, "readout", "head")}
                _save_so2(model, ndir, rec)
                ag = self.gev.aux_gap(model)
                if ag is not None:
                    rec["aux"] = {"gap": ag}
                I.atomic_write_json(ndir / "result.json", rec)
            self._lineage({"cand_key": rec["cand_key"], "node_id": node, "parent": None, "op": "seed",
                           "source": src.get("label"), "kappa_val_orig": rec["kappa_val_orig"], "delta_kappa": None,
                           "preservation_rel_l2": None, "params": rec["params"], "flops_2mac": rec["flops_2mac"],
                           "band": rec["band"], "accepted": True, "fid": 0})
            m = {k: rec[k] for k in ("node_id", "band", "kappa_val_orig", "params", "flops_2mac", "checkpoint",
                                     "checkpoint_sha256", "fid", "parent", "theta")}
            m["source"] = src.get("label")
            mem = self.state["members"][rec["band"]]
            if not any(x["node_id"] == node for x in mem):
                mem.append(m)
                self._submit_test(m)
        k = int((self.g.get("seeds") or {}).get("top_k", 4))
        for b in BANDS:                         # sementes: só as top-k ORIG-val por faixa ficam como pais
            mem = self.state["members"][b]
            seeds = [x for x in mem if x["fid"] == 0]
            keep = {x["node_id"] for x in pareto_order(seeds)[:k]}
            self.state["members"][b] = [x for x in mem if x["fid"] != 0 or x["node_id"] in keep]
        self.state["seeds_done"] = True
        self._save()

    # ---- um job (com OOM 10x300 s e adoção)
    def _job(self, parent: dict, ops: List[dict], theta, fid: int, node: str, seed: int, ckey: str,
             resume_from: Optional[dict] = None, genes: Optional[dict] = None) -> dict:
        ndir = self.out / "nodes" / node / f"f{fid}"
        done = I.read_json(ndir / "result.json")
        oc = self.g.get("oom") or {}
        retry_of = None
        if done is not None and I.adoptable(done, ckey=ckey, fitness_sig=self.gev.fitness_sig) is None:
            if not (oc.get("retry_failed") and done.get("status") == "failed" and done.get("reason") == "oom"):
                done["adopted"] = True
                return done
            # oom.retry_failed: OOM de contenção re-roda; o result.json antigo é arquivado, nunca apagado.
            # retry_of (= sufixo do arquivo) vai no result e abre um registro NOVO no lineage (mesma cand_key)
            retry_of = time.strftime('%Y%m%d_%H%M%S')
            (ndir / "result.json").rename(ndir / f"result.json.oom_{retry_of}")
        retries, sleep_s = int(oc.get("retries", 10)), float(oc.get("sleep_s", 300))
        for attempt in range(retries + 1):
            try:
                out = (self._run_job(parent, ops, theta, fid, node, seed, ckey, ndir, genes=genes) if resume_from is None
                       else self._run_resume_job(parent, resume_from, fid, node, seed, ckey, ndir, genes=genes))
                if retry_of:
                    out["retry_of"] = retry_of
                    I.atomic_write_json(ndir / "result.json", _jsonable(out))
                if os.environ.get("GROWTH_CRASH") == f"{node}:f{fid}":      # SÓ TESTE: kill -9 real
                    import signal
                    os.kill(os.getpid(), signal.SIGKILL)
                return out
            except PreservationViolated as exc:
                rec = {"status": "failed", "reason": "preservation", "error": str(exc)}
                break
            except SpifilFallback as exc:            # growth v3: init SEMPRE SPiFiL
                rec = {"status": "failed", "reason": "spifil_fallback", "decision": "infeasible_spifil",
                       "error": str(exc)}
                break
            except FV.NonFiniteFitness as exc:
                rec = {"status": "failed", "reason": "nan", "error": str(exc)}
                break
            except FV.RulerViolation:
                raise
            except Exception as exc:
                if _is_oom(exc):                  # growth_fix5 (3): não cabe nem com a GPU vazia -> sem retries
                    cap, info = oom_is_capacity(exc, total_gib=_gpu_total_gib(self.gev.device),
                                                margin_gib=float(_mem(self.g).get("oom_capacity_margin_gib", 1.0)))
                    if cap:
                        print(f"[growth] OOM de capacidade em {node} f{fid}: failed na hora (sem retries): "
                              f"{str(exc)[:300]}", flush=True)
                        rec = {"status": "failed", "reason": "oom_capacidade", "oom": info,
                               "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:]}
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        break
                if _is_oom(exc) and attempt < retries:
                    print(f"[growth] OOM em {node} f{fid} ({attempt + 1}/{retries}); espera {sleep_s:.0f} s: "
                          f"{str(exc)[:300]}", flush=True)
                    traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    time.sleep(sleep_s)
                    continue
                rec = {"status": "failed", "reason": "oom" if _is_oom(exc) else "error",
                       "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:]}
                break
        rec.update(node_id=node, cand_key=ckey, fitness_sig=self.gev.fitness_sig, fid=fid, parent=parent["node_id"],
                   ops=ops, theta=theta, kappa_val_orig=None, genes=genes or {}, **({"retry_of": retry_of} if retry_of else {}))
        if resume_from is not None:
            rec["promotion"] = {"mode": "resume", "from_fid": resume_from.get("fid"),
                                "from_cand_key": resume_from.get("cand_key")}
        I.atomic_write_json(ndir / "result.json", rec)
        return rec

    def _run_job(self, parent, ops, theta, fid, node, seed, ckey, ndir, genes: Optional[dict] = None) -> dict:
        t0 = time.time()
        g = apply_genes(self.g, genes or {})      # growth v3: genes do filho (ssl/aug/whiten/softhebb)
        dev = self.gev.device
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
        pmodel, _ = BB_load(parent["checkpoint"], parent["checkpoint_sha256"])
        # growth_fix4 (b): o pai é re-pontuado AQUI, na mesma leitura (fp32) e no mesmo processo do filho
        psv, parent_rulers = self.gev.score_parent(pmodel, parent["checkpoint_sha256"])   # growth_fix7: cache
        parent_rescored = float(psv["kappa"])
        pmodel.cpu()                             # o score levou o pai para a GPU; o realize_plan monta na CPU
        nfi = NewFilterInit(self.gev, self.g["init_method"], self.g.get("init_config") or {},
                            int(g.get("spifil_images_per_class", 4)), seed)
        if _v3(g).get("spifil_n_superpixels"):
            nfi.spifil_over["n_superpixels"] = int(_v3(g)["spifil_n_superpixels"])
        nfi.dwsep_spifil = bool(_v3(g).get("spifil_dwsep"))          # v3.1 bloco 6
        res = realize_plan(pmodel, ops, nfi, g)
        if _v3(g).get("spifil_strict"):
            bad = [r for r in nfi.log if r.get("fallback") or int(r.get("n_pytorch", 0) or 0) > 0]
            if bad:
                raise SpifilFallback(f"filtros novos sem SPiFiL: {json.dumps(bad, default=str)[:600]}")
        px = self.gev.ev.val_x[: int(self.g.get("preservation_images", 8))].float()
        pre = getattr(res, "v3_pre_replace", None)
        anchor_state, v3_rep = None, None
        if pre is not None:                      # growth v3 (4f): preservação medida no plano EXATO (pré-replace);
            gs_pre = MO.finalize(pre, px, data={"split": "validation", "n": int(px.shape[0])},   # o filho real
                                 tol=float(self.g.get("preservation_tol", 1e-5)), protect=self.g["protect"])  # muda a
            if gs_pre.preservation.verdict == "violated":                                          # função (WiSE)
                raise PreservationViolated(f"rel-L2 emb (pré-replace) {gs_pre.preservation.emb_rel_l2}")
            gs = MO.finalize(res, px, data={"split": "validation", "n": int(px.shape[0])},
                             tol=float(self.g.get("preservation_tol", 1e-5)), protect=self.g["protect"])
            v3_rep = {"function_preserving": False, "replace": getattr(res, "v3_replace", None),
                      "child_vs_parent_rel_l2_emb": gs.preservation.emb_rel_l2,
                      "child_vs_parent_verdict": gs.preservation.verdict}
            gs.preservation = gs_pre.preservation
            anchor_state = {k: v.detach().cpu().clone() for k, v in pre.child.state_dict().items()}
            del pre
        else:
            gs = MO.finalize(res, px, data={"split": "validation", "n": int(px.shape[0])},
                             tol=float(self.g.get("preservation_tol", 1e-5)), protect=self.g["protect"])
        pres = asdict(gs.preservation)
        if gs.preservation.verdict == "violated":
            raise PreservationViolated(f"rel-L2 emb {gs.preservation.emb_rel_l2}")
        child_band = band_of(res.child.spec.cost.params_total)
        if res.child.spec.cost.params_total > int(self.g.get("hard_cap", HARD_CAP)):
            raise BB.BudgetError("teto duro")
        del pmodel                               # growth_mem: o pai não fica vivo durante o treino do filho
        _free(dev, _mem(self.g)["empty_cache"])
        promo = _promo(self.g)
        want_resume = (int(fid) == int(self.g["fid_low"]) and int(self.g["fid_high"]) != int(self.g["fid_low"])
                       and promo["mode"] == "resume" and not ssl_disabled(g))
        cap = {} if want_resume else None
        sh_log, rep_log = getattr(res, "v3_softhebb", None), getattr(res, "v3_replace", None)
        model, out = train_child(self.gev, res, fid, g, seed, parent_rescored, capture=cap, anchor_state=anchor_state)
        bt = _v3(g).get("accept_bootstrap")       # v3.1 bloco 6: margem estatística (bootstrap pareado da val)
        cp, ct = out.pop("_val_pred", None), out.pop("_val_true", None)
        if bt and cp is not None and psv.get("pred") is not None:
            out["bootstrap"] = bootstrap_delta_kappa(np.asarray(psv["pred"]), np.asarray(cp), np.asarray(ct),
                                                     n=int(bt.get("n", 1000)), alpha=float(bt.get("alpha", 0.05)),
                                                     seed=int(seed))
        out.update(genes=genes or {}, softhebb_log=sh_log, replace_log=rep_log, prune_log=getattr(res, "v3_prune", None),
                   spifil_data={"images_per_class_requested": g.get("spifil_images_per_class"),
                                "images_per_class_effective": nfi.ipc_effective, "n_images": nfi.n_images,
                                "split": "train (pool estratificado semeado)",
                                "n_superpixels": nfi.spifil_over.get("n_superpixels", "padrão do cfg")})
        if v3_rep is not None:
            out["v3_replace"] = v3_rep
        if want_resume:                          # growth_fix6: pesos PRÉ-WiSE + AdamW/RNG para a promoção
            info = save_resume_state(ndir / RESUME_FILE, cap, save_optimizer=bool(promo["save_optimizer"]),
                                     meta={"node_id": node, "fid": int(fid), "cand_key": ckey, "seed": int(seed),
                                           "parent": parent["node_id"]})
            out.update(resume_state=info["path"], resume_state_sha256=info["sha256"],
                       resume_state_has_optimizer=info["has_optimizer"], resume_state_has_rng=info["has_rng"])
            del cap
        model.spec.provenance.parent = parent["node_id"]
        model.spec.provenance.wise_alpha = out["wise_alpha"]
        model.spec.provenance.fidelity_epochs = int(fid)
        ck = BB_save(model, ndir / "backbone.pt")
        _save_so2(model, ndir, out)
        (ndir / "growth_spec.json").write_text(gs.to_json())
        peak = (torch.cuda.max_memory_reserved(dev) / 1e9) if dev.type == "cuda" else None
        out.update(node_id=node, cand_key=ckey, fitness_sig=self.gev.fitness_sig, fid=int(fid),
                   parent=parent["node_id"], parent_checkpoint_sha256=parent["checkpoint_sha256"], ops=ops,
                   theta=theta, seed=int(seed), band=child_band, params=int(model.spec.cost.params_total),
                   flops_2mac=int(model.spec.cost.flops), flops=int(model.spec.cost.macs), arch=model.spec.id,
                   checkpoint=ck["path"], checkpoint_sha256=ck["file_sha256"], growth_spec_id=gs.id,
                   preservation=pres, preservation_rel_l2_emb=gs.preservation.emb_rel_l2,
                   preservation_verdict=gs.preservation.verdict, new_filter_init=nfi.log,
                   protect=self.g["protect"], init_method=self.g["init_method"],
                   delta_kappa=out["kappa_val_orig"] - parent_rescored,
                   kappa_parent_rescored=parent_rescored, parent_kappa_val_orig_member=float(parent["kappa_val_orig"]),
                   parent_rescore_rulers=parent_rulers,
                   wall_s=time.time() - t0, peak_gpu_reserved_gb=peak,
                   inheritance_totals=gs.inheritance.get("totals"))
        I.atomic_write_json(ndir / "result.json", _jsonable(out))
        return out

    def _run_resume_job(self, parent, r_low, fid, node, seed, ckey, ndir, genes: Optional[dict] = None) -> dict:
        """growth_fix6: promoção por RETOMADA. Carrega o filho de f_low (pesos PRÉ-WiSE + AdamW/RNG), treina as
        épocas restantes, WiSE com a âncora = morfismo, e fica com o melhor de f_low/f_high pela ORIG-val."""
        t0 = time.time()
        dev = self.gev.device
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
        state = load_resume_state(r_low["resume_state"], r_low.get("resume_state_sha256"))
        pmodel, _ = BB_load(parent["checkpoint"], parent["checkpoint_sha256"])
        self.gev.reset_stamps()                  # growth_fix4 (b): pai re-pontuado no mesmo processo/leitura
        psv = self.gev.score(pmodel)
        parent_rescored = float(psv["kappa"])
        parent_rulers = self.gev.decision_rulers()
        del pmodel
        _free(dev, _mem(self.g)["empty_cache"])
        model, _ = BB_load(r_low["checkpoint"], r_low["checkpoint_sha256"])   # arquitetura do filho de f_low
        model.load_state_dict(state["trained"])                                 # pesos treinados PRÉ-WiSE
        fl = int(r_low["fid"])
        n_extra = resume_extra_epochs(fl, int(fid), self.g, self.cfg.get("train"))
        g = apply_genes(self.g, genes or {})
        model, out = continue_child(self.gev, model, state, n_extra, g, seed, parent_rescored)
        out["genes"] = genes or {}
        del state
        model.spec.provenance.parent = parent["node_id"]
        model.spec.provenance.wise_alpha = out["wise_alpha"]
        model.spec.provenance.fidelity_epochs = int(fid)
        ck_high = BB_save(model, ndir / "backbone.pt")
        _save_so2(model, ndir, out)
        low_dir = Path(r_low["checkpoint"]).parent
        if (low_dir / "growth_spec.json").exists():
            (ndir / "growth_spec.json").write_text((low_dir / "growth_spec.json").read_text())
        k_low, a_low = float(r_low["kappa_val_orig"]), float(r_low["wise_alpha"])
        k_high, a_high = float(out["kappa_val_orig"]), float(out["wise_alpha"])
        pick = best_of_low_high(k_low, k_high)                 # seleção pela VALIDAÇÃO (ORIG-val), nunca teste
        out.update(kappa_val_orig_f_low=k_low, wise_alpha_f_low=a_low, kappa_val_orig_f_high=k_high,
                   wise_alpha_f_high=a_high, fid_low=fl, fid_high=int(fid),
                   aux_f_low={"wise_kappas": (r_low.get("aux") or {}).get("wise_kappas")},
                   checkpoint_f_high=ck_high["path"], checkpoint_sha256_f_high=ck_high["file_sha256"],
                   selected_fid=int(fid) if pick else fl,
                   selection="best_of_low_high: kappa ORIG-val; f_high salvo kappa_val(f_high) < kappa_val(f_low) "
                             "(empate = f_high); teste nunca lido",
                   promotion={"mode": "resume", "from_fid": fl, "from_cand_key": r_low["cand_key"],
                              "extra_ssl_epochs": int(n_extra), "extra_epochs_rule": _promo(self.g)["extra_epochs"],
                              **out.pop("resume")})
        if pick:
            ck = ck_high
        else:                                    # f_low venceu pela val: o nó fica com o checkpoint de f_low
            ck = {"path": r_low["checkpoint"], "file_sha256": r_low["checkpoint_sha256"]}
            out.update(kappa_val_orig=k_low, quality=FV.quality_of(k_low), wise_alpha=a_low)
            # HOTFIX (fix6): o nó é o de f_low -> os kappas do WiSE que justificam alpha/kappa são os de f_low
            # (antes ficavam os de f_high e o check_decisions levantava RulerViolation FATAL quando alpha_low não era
            # o argmax dos kappas de f_high). Os de f_high continuam registrados em aux_f_high.
            aux_low = dict(r_low.get("aux") or {})
            out["aux_f_high"] = {"wise_kappas": (out.get("aux") or {}).get("wise_kappas"),
                                 "kappa_val_alpha0": (out.get("aux") or {}).get("kappa_val_alpha0")}
            out["aux"] = {**(out.get("aux") or {}), "wise_kappas": aux_low.get("wise_kappas"),
                          "kappa_val_alpha0": aux_low.get("kappa_val_alpha0"),
                          "macro_f1": aux_low.get("macro_f1", (out.get("aux") or {}).get("macro_f1"))}
            for k in ("so2_head_sha256", "so2_head_meta", "so2_head"):       # growth v3: cabeça do f_low
                if r_low.get(k) is not None:
                    out[k] = r_low[k]
        peak = (torch.cuda.max_memory_reserved(dev) / 1e9) if dev.type == "cuda" else None
        out.update(node_id=node, cand_key=ckey, fitness_sig=self.gev.fitness_sig, fid=int(fid),
                   parent=parent["node_id"], parent_checkpoint_sha256=parent["checkpoint_sha256"], ops=r_low["ops"],
                   theta=r_low["theta"], seed=int(seed), band=r_low["band"], params=int(model.spec.cost.params_total),
                   flops_2mac=int(model.spec.cost.flops), flops=int(model.spec.cost.macs), arch=model.spec.id,
                   checkpoint=ck["path"], checkpoint_sha256=ck["file_sha256"], growth_spec_id=r_low.get("growth_spec_id"),
                   preservation=r_low.get("preservation"), preservation_rel_l2_emb=r_low.get("preservation_rel_l2_emb"),
                   preservation_verdict=r_low.get("preservation_verdict"), new_filter_init=r_low.get("new_filter_init"),
                   protect=self.g["protect"], init_method=self.g["init_method"],
                   delta_kappa=out["kappa_val_orig"] - parent_rescored,
                   kappa_parent_rescored=parent_rescored, parent_kappa_val_orig_member=float(parent["kappa_val_orig"]),
                   parent_rescore_rulers=parent_rulers,
                   wall_s=time.time() - t0, peak_gpu_reserved_gb=peak,
                   inheritance_totals=r_low.get("inheritance_totals"))
        FV.check_decisions(out, self.gev.fitness_spec)
        I.atomic_write_json(ndir / "result.json", _jsonable(out))
        return out

    def _promote(self, r_low: dict, meta: dict, fl: int, fh: int) -> dict:
        """growth_fix6: f_low -> f_high. ``promotion.mode=resume`` (padrão) retoma o filho de f_low quando o
        estado de retomada existe (íntegro); senão (ou ``retrain``/``ssl: none``) refaz do morfismo (histórico),
        com o motivo registrado. A cand_key da retomada inclui a cand_key de f_low (não adota um f_high refeito)."""
        promo = _promo(self.g)
        why = None
        if promo["mode"] != "resume":
            why = "promotion.mode=retrain"
        elif ssl_disabled(apply_genes(self.g, meta.get("genes") or {})):
            why = "ssl: none (não há SSL para continuar)"
        elif not r_low.get("resume_state"):
            why = "f_low sem resume_state (resultado anterior ao fix6)"
        elif I.file_sha256(r_low["resume_state"]) != r_low.get("resume_state_sha256"):
            why = "resume_state ausente ou com sha256 diferente"
        if why is None:
            n_extra = resume_extra_epochs(fl, fh, self.g, self.cfg.get("train"))
            rkey = I.lineage_key({"resume_from": r_low["cand_key"], "extra_ssl_epochs": n_extra,
                                  "format": RESUME_FORMAT}, meta["seed"], {"kind": "resume", "lkey": meta["lkey"]},
                                 self.gev.fitness_sig)
            r = self._job(meta["parent"], meta["ops"], meta["theta"], fh, meta["node"], meta["seed"],
                          I.cand_key(rkey, fh), resume_from=r_low, **_gk(meta.get("genes")))
            return select_low_high(r_low, r, fl, fh)
        print(f"[growth] {meta['node']}: promoção f{fl}->f{fh} REFEITA do morfismo ({why})", flush=True)
        r = self._job(meta["parent"], meta["ops"], meta["theta"], fh, meta["node"], meta["seed"],
                      I.cand_key(meta["lkey"], fh), **_gk(meta.get("genes")))
        r = dict(r)
        r["promotion"] = {**(r.get("promotion") or {}), "mode": "retrain", "reason": why}
        return select_low_high(r_low, r, fl, fh)      # HOTFIX member_fid: melhor de f_low/f_high também no retrain

    # ---- teste (canal separado; nunca lido aqui)
    def _submit_test(self, member: dict):
        with self.test_queue.open("a") as f:
            f.write(json.dumps({"node_id": member["node_id"], "checkpoint": member["checkpoint"],
                                "checkpoint_sha256": member["checkpoint_sha256"], "band": member["band"],
                                "fid": member["fid"], "never_feeds_fitness": True}) + "\n")

    def _run_test_channel(self):
        if not self.g.get("test_channel", True):
            return
        cmd = [self.python, "-m", "ga_search.hmodehb.growth_test_channel", "--out", str(self.out)]
        try:
            subprocess.run(cmd, check=False, timeout=6 * 3600)
        except Exception as exc:             # o canal de teste nunca derruba a busca
            print(f"[growth] canal de teste falhou: {exc}", flush=True)

    # ---- uma geração
    def _generation(self, band: str, gen: int):
        tag = f"{band}:{gen}"
        if tag in self.state["gens_done"]:
            return
        target = BANDS[BANDS.index(band) + 1]
        parents = pareto_order([m for m in self.state["members"][band]])[: int(self.g["n_parents"])]
        if not parents:
            print(f"[growth] {band}: sem pais", flush=True)
            self.state["gens_done"].append(tag)
            self._save()
            return
        pool = [m["theta"] for m in self.state["members"][target] if m.get("theta")]
        rng = np.random.default_rng(_h({"tag": tag, "seed": self.cfg["seed"]}))
        fl, fh = int(self.g["fid_low"]), int(self.g["fid_high"])
        low: List[Tuple[dict, dict]] = []
        nch = int(self.g["n_children"])
        thetas = [de_theta(pool, rng) for _ in range(nch)]
        pseeds = [_h({"tag": tag, "i": i, "plan": True}) % (2 ** 31 - 1) for i in range(nch)]
        pm = [(m, BB_load(m["checkpoint"], m["checkpoint_sha256"])[0]) for m in parents]
        mctx = mem_ctx_from(self.cfg, self.g, self.gev.device)
        genes_l = [child_genes(self.g, tag, i, int(self.cfg["seed"])) for i in range(nch)]   # growth v3: genes
        plans = plan_children(pm, target, nch, thetas, pseeds, self.g, int(self.g.get("hard_cap", HARD_CAP)),
                              mem_ctx=mctx, gcfgs=[apply_genes(self.g, gg) for gg in genes_l])
        del pm
        for i, pl in enumerate(plans):
            parent, ops, theta = pl["parent"], pl["ops"], pl["theta"]
            seed = _h({"tag": tag, "i": i, "parent": parent["checkpoint_sha256"]}) % (2 ** 31 - 1)
            node = f"{target}g{gen}c{i:02d}"
            for k, inf in enumerate(pl.get("infeasible_mem") or []):     # growth_fix5: registrado e substituído
                ikey = I.lineage_key({"ops": inf["ops"], "theta": inf["theta"], "infeasible_mem": True}, inf["seed"],
                                     {"parent_sha256": inf["parent"]["checkpoint_sha256"]}, self.gev.fitness_sig)
                print(f"[growth] {node}: {inf['reason']}", flush=True)
                self._lineage({"cand_key": I.cand_key(ikey, 0), "node_id": f"{node}_mem{k}",
                               "parent": inf["parent"]["node_id"], "op": "+".join(o["op"] for o in inf["ops"]),
                               "ops": inf["ops"], "family": inf["family"], "accepted": False, "status": "infeasible_mem",
                               "decision": "infeasible_mem", "reason": inf["reason"],
                               "mem_estimate": _jsonable(inf["mem_estimate"]), "band": target})
            origin = {"kind": "inherit", "parent_sha256": parent["checkpoint_sha256"], "protect": self.g["protect"],
                      "init": self.g["init_method"], "init_config": self.g.get("init_config") or {}}
            genes = genes_l[i]
            if genes:                                # growth v3: os genes entram na cand_key (sem genes: chave histórica)
                origin["genes"] = genes
            lkey = I.lineage_key({"ops": ops, "theta": theta}, seed, origin, self.gev.fitness_sig)
            if ops is None:
                print(f"[growth] {node}: {pl['reason'][:500]}", flush=True)
                self._lineage({"cand_key": I.cand_key(lkey, 0), "node_id": node, "parent": parent["node_id"],
                               "op": None, "accepted": False, "decision": "no_plan", "reason": pl["reason"],
                               "band": target})
                continue
            r = self._job(parent, ops, theta, fl, node, seed, I.cand_key(lkey, fl), **_gk(genes))
            decision, ok = decide(r, self.g["delta"], self.g.get("require_improvement", False))
            self._lineage(self._lin_rec(r, parent, ops, fl, ok, final=(fl == fh) or decision == "adopted_as_parent",
                                        decision=decision, family=pl.get("family")))
            if ok:
                low.append((r, {"parent": parent, "ops": ops, "theta": theta, "seed": seed, "node": node, "lkey": lkey,
                                "family": pl.get("family"), "genes": genes}))
            self._save()
        if _v3(self.g).get("control_no_grow"):     # growth v3: controle LEMONADE (treinar mais SEM crescer)
            for p_i, parent in enumerate(parents):
                node = f"{target}g{gen}ctl{p_i:02d}"
                seed = _h({"tag": tag, "control": p_i, "parent": parent["checkpoint_sha256"]}) % (2 ** 31 - 1)
                genes = genes_l[p_i % len(genes_l)] if genes_l else {}
                origin = {"kind": "control_no_grow", "parent_sha256": parent["checkpoint_sha256"], "genes": genes}
                lkey = I.lineage_key({"ops": [], "control": True}, seed, origin, self.gev.fitness_sig)
                r = self._job(parent, [], None, fl, node, seed, I.cand_key(lkey, fl), genes=genes)
                rec = self._lin_rec(r, parent, [], fl, False, final=True, decision="control_no_grow",
                                    family="control_no_grow")
                rec.update(op="control_no_grow", accepted=False, used_for_selection="never (controle)")
                self._lineage(rec)
            self._save()
        # promoção: 1/eta melhores aceitos em fid_low (Pareto kappa x FLOPs) -> fid_high
        n_prom = max(1, int(math.ceil(len(low) / float(self.g.get("eta", 3))))) if low else 0
        ranked = pareto_order([{**r, "node_id": r["node_id"]} for r, _ in low])[:n_prom]
        chosen = {r["node_id"] for r in ranked}
        for r, meta in low:
            if r["node_id"] not in chosen:
                continue
            if fh != fl:
                r = self._promote(r, meta, fl, fh)         # growth_fix6: retoma f_low (ou refaz, registrado)
                decision, ok = decide(r, self.g["delta"], self.g.get("require_improvement", False))
                self._lineage(self._lin_rec(r, meta["parent"], meta["ops"], fh, ok, final=True, decision=decision,
                                            family=meta.get("family")))
                if not ok:
                    continue
            m = {k: r[k] for k in ("node_id", "band", "kappa_val_orig", "params", "flops_2mac", "checkpoint",
                                   "checkpoint_sha256", "fid", "parent", "theta")}
            if r.get("selected_fid") is not None and fh != fl:   # growth_fix6: qual fidelidade ficou (val)
                m["selected_fid"] = int(r["selected_fid"])
                m["fid"] = int(r["selected_fid"])                # hotfix member_fid: o checkpoint é o dessa fid
            mem = self.state["members"][target]
            if r["band"] != target or r["params"] > HARD_CAP:
                continue
            if not any(x["node_id"] == m["node_id"] for x in mem):
                mem.append(m)
                self._submit_test(m)
            self._save()
        self.state["gens_done"].append(tag)
        self._save()

    # ---- growth v3 (6): saturação -> o melhor membro da faixa anterior é ADOTADO; curva kappa x faixa
    def _saturate(self, band: str, target: str) -> None:
        if not (_v3(self.g).get("saturation") or {}).get("adopt_parent"):
            return
        tag = f"saturate:{band}->{target}"
        if tag in self.state.setdefault("saturation_done", []):
            return
        src = pareto_order(list(self.state["members"][band]))[:1]
        if src:
            best_src = max(self.state["members"][band], key=lambda m: (m["kappa_val_orig"], -m["params"]))
            tgt = self.state["members"][target]
            best_t = max((m["kappa_val_orig"] for m in tgt), default=None)
            if best_t is None or best_t < best_src["kappa_val_orig"]:
                m = {k: best_src[k] for k in ("band", "kappa_val_orig", "params", "flops_2mac", "checkpoint",
                                              "checkpoint_sha256", "fid", "parent", "theta")}
                m.update(node_id=f"{target}_adopt_{best_src['node_id']}", band=target, adopted_from=best_src["node_id"],
                         adopted_as_parent=True, parent=best_src["node_id"],
                         reason=("faixa sem filho aceito" if best_t is None else
                                 f"melhor da faixa {best_t:.4f} < melhor de {band} {best_src['kappa_val_orig']:.4f}"))
                if not any(x["node_id"] == m["node_id"] for x in tgt):
                    tgt.append(m)
                ikey = I.cand_key(I.lineage_key({"adopt": best_src["node_id"], "target": target}, 0,
                                                {"kind": "saturation"}, self.gev.fitness_sig), 0)
                self._lineage({"cand_key": ikey, "node_id": m["node_id"], "parent": best_src["node_id"],
                               "op": "adopt_parent", "decision": "adopted_parent_saturation", "accepted": True,
                               "kappa_val_orig": m["kappa_val_orig"], "params": m["params"], "band": target,
                               "reason": m["reason"], "delta_kappa": 0.0})
                print(f"[growth] saturação {band}->{target}: {best_src['node_id']} adotado ({m['reason']})", flush=True)
        self.state["saturation_done"].append(tag)
        self._save()

    def _curve(self) -> List[dict]:
        """Curva kappa ORIG-val x faixa (melhor membro por faixa; adotados marcados). Gravada em kappa_band_curve.json."""
        rows = []
        for b in BANDS:
            mem = self.state["members"].get(b) or []
            if not mem:
                rows.append({"band": b, "best_node": None, "kappa_val_orig": None, "n_members": 0})
                continue
            best = max(mem, key=lambda m: (m["kappa_val_orig"], -m["params"]))
            rows.append({"band": b, "best_node": best["node_id"], "kappa_val_orig": best["kappa_val_orig"],
                         "params": best["params"], "adopted": bool(best.get("adopted_as_parent", False)),
                         "n_members": len(mem), "readout": getattr(self.gev, "readout", "head")})
        I.atomic_write_json(self.out / "kappa_band_curve.json", {"curve": rows, "fitness_sig": self.gev.fitness_sig,
                                                                 "selection": "validation_only"})
        return rows

    def _lin_rec(self, r: dict, parent: dict, ops, fid, ok, final, decision=None, family=None) -> dict:
        pk = r.get("kappa_parent_rescored")
        return {"cand_key": r.get("cand_key"), "node_id": r.get("node_id"), "parent": parent["node_id"],
                "parent_checkpoint_sha256": parent["checkpoint_sha256"], "op": "+".join(o["op"] for o in ops),
                "ops": ops, "fid": fid, "status": r.get("status"), "reason": r.get("reason"),
                "decision": decision, "family": family,
                # growth_fix4 (b): os dois kappas da decisão (mesma leitura fp32, mesmo processo)
                "kappa_val_orig": r.get("kappa_val_orig"),
                "parent_kappa_val_orig": parent["kappa_val_orig"] if pk is None else pk,
                "kappa_parent_rescored": pk, "parent_kappa_val_orig_member": parent["kappa_val_orig"],
                "embed_precision": r.get("embed_precision"),
                "delta_kappa": r.get("delta_kappa"), "wise_alpha": r.get("wise_alpha"),
                "preservation_rel_l2": r.get("preservation_rel_l2_emb"),
                "preservation_verdict": r.get("preservation_verdict"), "params": r.get("params"),
                "flops_2mac": r.get("flops_2mac"), "band": r.get("band"), "accepted": bool(ok), "final": bool(final),
                "adopted": bool(r.get("adopted", False)), "used_for_selection": "validation_only",
                # growth v3: genes, leitura, DA, SSL efetivo, aux GAP, replace
                **({k: r.get(k) for k in ("genes", "readout", "ssl", "ssl_method", "reg", "aug", "aug_gene",
                                          "whiten_new", "softhebb", "anchor", "so2_head_sha256", "v3_replace", "retry_of")
                    if r.get(k) is not None}),
                **({"kappa_val_gap_aux": ((r.get("aux") or {}).get("gap") or {}).get("kappa_val_gap")}
                   if (r.get("aux") or {}).get("gap") else {}),
                # growth_fix6: promoção por retomada (alpha e kappa de f_low e f_high; qual ficou, pela val)
                **({k: r.get(k) for k in ("selected_fid", "kappa_val_orig_f_low", "wise_alpha_f_low",
                                          "kappa_val_orig_f_high", "wise_alpha_f_high", "promotion",
                                          "resume_state_has_optimizer") if r.get(k) is not None})}

    # ---- execução
    def run(self) -> dict:
        self.out.mkdir(parents=True, exist_ok=True)
        with I.ControllerLock(self.out):
            self.state = self._load()
            self._lin_keys = self._lineage_keys()
            self._save()
            if not self.state.get("seeds_done"):
                self._seed_stage()
            b0, b1 = BANDS.index(self.g.get("start_band", "B0")), BANDS.index(self.g.get("end_band", "B4"))
            for k in range(b0, b1):
                for gen in range(int(self.g["generations"])):
                    self._generation(BANDS[k], gen)
                self._saturate(BANDS[k], BANDS[k + 1])       # growth v3 (6)
                self._run_test_channel()
            curve = self._curve()
            summary = {"best_by_band": {b: (pareto_order(self.state["members"][b])[:1] or [None])[0] for b in BANDS},
                       "members": self.state["members"], "fitness_sig": self.gev.fitness_sig,
                       "kappa_band_curve": curve, "readout": getattr(self.gev, "readout", "head"),
                       "selection": "kappa ORIG-val (teste nunca lido pelo controlador)"}
            I.atomic_write_json(self.out / "summary.json", summary)
            return summary


def _gk(genes: Optional[dict]) -> dict:
    """kwarg ``genes`` só quando há genes (sem genes: a chamada histórica de ``_job``)."""
    return {"genes": genes} if genes else {}


def _save_so2(model, ndir: Path, out: dict) -> None:
    """growth v3 (1): a cabeça de 2ª ordem fixa ajustada (buffers mu/V; PCA só no treino) ao lado do checkpoint."""
    h = getattr(model, "so2_head", None)
    if h is None:
        return
    p = Path(ndir) / "so2_head.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(h.payload(), p)
    out["so2_head"] = str(p)
    out["so2_head_sha256"] = h.sha256()


# ═════════════════════════════════════════════════════════════════ checkpoint helpers
def BB_save(model: BB.Backbone, path) -> dict:
    from ga_search.backbones import checkpoint as CK
    try:
        return CK.save_backbone(model, path, torchscript=True)
    except Exception:                        # trace falhou (op não traçável): salva sem TorchScript
        return CK.save_backbone(model, path, torchscript=False)


def BB_load(path, expected_sha256=None):
    from ga_search.backbones import checkpoint as CK
    return CK.load_backbone(path, expected_sha256=expected_sha256)


def _is_backbone_ckpt(path) -> bool:
    try:
        import zipfile
        with zipfile.ZipFile(path) as z:      # sem desserializar: procura a string do formato no pickle
            return any(n.endswith("data.pkl") and b"backbone_ckpt/v1" in z.read(n) for n in z.namelist())
    except Exception:
        return False


def _jsonable(x):
    return json.loads(json.dumps(x, default=str))


# ═════════════════════════════════════════════════════════════════ CLI
def load_growth_config(path=None, overrides=()) -> dict:
    g = copy.deepcopy(GROWTH_DEFAULTS)
    p = Path(path) if path else GROWTH_CONFIG
    if p.exists():
        raw = yaml.safe_load(p.read_text()) or {}
        for k, v in raw.items():
            g[k] = {**g[k], **v} if isinstance(g.get(k), dict) and isinstance(v, dict) else v
    for item in overrides or ():
        from ga_search.modehb.run_search import _set_dotted
        _set_dotted(g, item)
    if g["protect"] not in PROTECT_SCALE:
        raise ValueError(f"protect {g['protect']!r}")
    if int(g.get("hard_cap", HARD_CAP)) > HARD_CAP:
        raise ValueError("hard_cap acima de 5,5M é proibido")
    return g


def build_cfg(dataset: str, seed: int, overrides=(), out_dir=None) -> dict:
    from ga_search.hmodehb import run_hsearch as R
    cfg = R.load_hconfig("A", dataset, seed, list(overrides))
    cfg["output_dir"] = str(out_dir or f"artifacts/hmodehb_b0b4/growth/{dataset}/seed{seed}")
    cfg["hmodehb"]["kind"] = KIND
    return cfg


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="cadeia de crescimento B0->B4 (SPiFiL preservado, aceitação ORIG-val)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--growth-config", default=None)
    ap.add_argument("--set", action="append", default=[], help="override do config de treino/dados (a.b=V)")
    ap.add_argument("--gset", action="append", default=[], help="override do config de crescimento (a.b=V)")
    ap.add_argument("--allow-drift", action="append", default=[])
    ap.add_argument("--cov-eigh-workers", type=int, default=None)
    return ap.parse_args(argv)


def main(argv=None):
    from ga_search.modehb.run_search import _cap_cpu_threads
    import faulthandler
    import signal
    # growth_fix7: `kill -USR1 <pid>` despeja a pilha Python de TODAS as threads no log (stderr) sem parar o
    # processo -> diagnóstico de fase "parada" sem py-spy/ptrace (athena).
    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    a = parse_args(argv)
    ov = list(a.set) + ["wandb.enabled=false"]
    if a.device:
        ov.append(f"device={a.device}")
    if a.cov_eigh_workers is not None:
        ov.append(f"eval.svm_fitness.cov_eigh_workers={a.cov_eigh_workers}")
    cfg = build_cfg(a.dataset, a.seed, ov, a.out)
    g = load_growth_config(a.growth_config, a.gset)
    try:
        _cap_cpu_threads()
    except Exception:
        pass
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    gev = GrowthEvaluator(cfg, out / "eval", g.get("debug_train"), g.get("debug_val"), v3=g.get("v3"))
    chain = GrowthChain(cfg, g, gev, out, allow_drift=a.allow_drift)
    try:
        s = chain.run()
    except I.ControllerLocked as exc:
        print(f"[growth] {exc}", flush=True)
        return 3
    except I.ConfigDrift as exc:
        print(f"[growth] {exc}", flush=True)
        return 4
    print(json.dumps({b: (v or {}).get("kappa_val_orig") for b, v in s["best_by_band"].items()}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
