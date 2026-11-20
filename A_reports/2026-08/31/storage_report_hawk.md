# Relatório de Armazenamento — Repositório Local
**Escopo:** somente `/dados/home/moliveira/scalable_FLIM_self_supervised` (sem acesso a cluster remoto — ver nota abaixo)
**Data:** 2026-07-07
**Usuário:** moliveira

> Nota: o prompt original pedia investigação via SSH no cluster "Hawk". Não há configuração de SSH funcional para esse host neste ambiente (`ssh hawk` falha em resolver o hostname), então a investigação foi restrita a este repositório local, conforme instruído.

## 1. Resumo de Uso Total
| Filesystem | Ponto de montagem | Usado | Total | Disponível | % |
|---|---|---|---|---|---|
| `/dev/mapper/dados--vg-lv--dados` | `/dados` | 2.3T | 2.9T | 457G | 84% |
| **Este repositório** | `.` | **1.4T** | — | — | ~61% do `/dados` usado |

O repositório sozinho responde por **1.4 TB**, quase todo dentro de `artifacts/distillation/` (1.4T de 1.4T).

## 2. Top Diretórios por Tamanho
| Caminho | Tamanho |
|---|---|
| `artifacts/distillation/` | 1.4T |
| `logs/` (todo) | 21G |
| `logs/flim-ssl/` | 15G |
| `logs/wandb/` | 5.5G |
| `wandb/` (1798 run-dirs) | 1.4G |
| `data/` | 639M |
| `artifacts/SVM/` | 188M |
| `.claude/worktrees/` | 102M |
| `artifacts_view/MLP/` | 46M |
| `.git/` | 26M |
| `artifacts/plots/` | 18M |
| `Presentation_FLIM_unsupervised/images/` | 17M |
| `FAPESP_PhD_hawk/` | 15M |
| `graphify-out/` | 9.7M |

`540` diretórios de run existem sob `artifacts/distillation/`.

## 3. Arquivos Grandes (>500 MB)
Dominados por checkpoints `.ckpt` de distillation, quase todos na faixa **2.38–2.52 GB** cada:

| Padrão | Tamanho típico | Ocorrências |
|---|---|---|
| `checkpoints/last.ckpt`, `best-epoch=*.ckpt` (proj-types antigos) | 2.38–2.52 GB | 561 arquivos |
| `checkpoints/*.ckpt` (proj-types com `FrozenTeacherCheckpointMixin`) | ~10–30 MB | 253 arquivos |

Um run (`distillation_eggs_split2_pct100_2l_1x1_init_flim_256_1280`) acumulou **3** checkpoints "best" de épocas diferentes (049 e 092) além do `last.ckpt` → 7.2 GB só nesse diretório, em vez dos ~4.8 GB dos runs equivalentes.

## 4. Achado Principal — Bloat do Teacher Congelado em Checkpoints Antigos

Confirmado no código (`src/models/distillation.py:136-158`, `FrozenTeacherCheckpointMixin`): o teacher I-JEPA congelado (~2.5 GB) é removido do `state_dict` ao salvar e **reinjetado automaticamente** a partir do módulo vivo ao carregar (`on_load_checkpoint`), porque é sempre o mesmo peso pré-treinado, não específico do run.

Esse mixin só é usado por `DistillationOneLayerModule` e `DistillationTwoLayerModule` (`src/modules/distillation_onelayer_module.py`, `distillation_twolayer_module.py`). As classes mais antigas — `DistillationModule` e `DistillationConvModule` (`src/modules/distillation_module.py`, `distillation_conv_module.py`) — **não herdam o mixin**, mas usam exatamente os mesmos prefixos de `state_dict` (`student.`, `proj_kd.`, `teacher.`), confirmado inspecionando um checkpoint real:

```
top-level prefixes: {'proj_kd', 'teacher', 'student'}
teacher size: 2.52 GB  |  total state_dict: 2.55 GB   (99% do arquivo é o teacher)
```

**Resultado:** 561 arquivos `.ckpt` de proj-types antigos (`modeldirect`, `next_layers_direct`, `3x3_BN2d_1280_one_layer[_init_flim]`, `2l_1x1_init_flim_256_1280`, `2l_1x1_BN2d_256_1280`, `1x1_BN2d_1280_one_layer[_flim_init]`, `1x1_BN2d_1280_one_layer_flim_init_no_imagenet_norm`) carregam o teacher inteiro = **1330.8 GB**, contra **6.87 GB** dos 253 checkpoints já "stripados" pelos módulos novos.

Status dos 540 runs com checkpoint bloatado (via `run_metadata.json`):
| Status | Diretórios |
|---|---|
| `ok` (treino concluído) | 122 |
| `error` | 4 |
| sem `run_metadata.json` (legado, anterior ao tracking) | 404 |

Como o `state_dict` usa os mesmos prefixos, o mesmo filtro (`keep only student.*, proj_kd.*`) aplicado post-hoc é **seguro para inferência/SVM** (que só usa `student.encode()` ou a projeção) em qualquer run já concluído. O único risco é **resumir treino** de um run que ainda não tenha o `on_load_checkpoint` de reinjeção — isso afeta só os 4 diretórios com `status=error` (potencialmente retomáveis) e, por precaução, os 404 sem metadata caso algum ainda esteja ativo.

## 5. Outros Candidatos Menores

### 5.1 PODE APAGAR com segurança (recriável/temporário)
| Caminho | Tamanho | Motivo |
|---|---|---|
| `__pycache__/` (14 dirs) | 1.2 MB | Bytecode recriado automaticamente |
| `*.pyc` soltos (75 arquivos) | incluído acima | idem |
| `.ipynb_checkpoints` | 0 (nenhum encontrado) | — |
| `core.*`, `*.tmp`, `*.swp` | 0 (nenhum encontrado) | — |

Valor total: **desprezível (~1-2 MB)** — não é onde está o problema deste repo.

### 5.2 AVALIAR ANTES DE APAGAR
| Caminho | Tamanho | Motivo |
|---|---|---|
| **1330.8 GB de teacher bloat em `artifacts/distillation/*/checkpoints/*.ckpt`** | 1.33 TB | Ver seção 4 — stripar (não apagar o arquivo), preservando `student.*`/`proj_kd.*` |
| `*.log` soltos (≤ profundidade 6) | 85 MB | Confirmar se são logs de jobs já finalizados |
| `wandb/` (1798 run-dirs) + `logs/wandb/` | 1.4G + 5.5G | **Não apagar sem checar sync**: CLAUDE.md indica que muitos runs de distillation são *offline* (não sincronizados no W&B cloud) — esses diretórios locais podem ser a única cópia das curvas de métrica |
| `logs/flim-ssl/` (787 subdirs) | 15G | Logs/checkpoints de runs LeJEPA SSL — checar quais splits/pcts ainda são referenciados pelas análises em `analysis_flim_distill/` antes de podar |
| Diretórios de proj-types legados sem `run_metadata.json` (404 runs) | incluído nos 1.33 TB | Confirmar com o usuário se `modeldirect`/`next_layers_direct`/`3x3_BN2d_*` etc. ainda são comparados nos plots atuais ou se só `2l_1x1_init_flim_256_1280` e a variante `_no_imagenet_norm` seguem ativos — se legado, dá para **apagar o diretório inteiro**, não só stripar |

### 5.3 NÃO APAGAR (insubstituível)
| Caminho | Tamanho | Motivo |
|---|---|---|
| `data/to_mateus/` | 262 MB | Pesos FLIM pré-treinados + `architecture.json` — input do pipeline, não recriável localmente |
| `data/to_modules/`, `data/reports_felipe/` | 227M + 146M | Dados/relatórios externos recebidos |
| Código-fonte (`src/`, `scripts/`, `configs/`) | — | Insubstituível sem backup |
| `artifacts/plots/`, `results/*.csv` | 18M | Resultados finais normalizados que alimentam as figuras |
| `analysis_flim_distill/*.md` | — | Investigações com evidência acumulada, referenciadas no CLAUDE.md |
| `.git/` | 26M | Histórico do projeto |

## 6. Estimativa de Espaço Recuperável
| Categoria | Espaço Estimado |
|---|---|
| Caches/pyc (5.1) | ~1-2 MB |
| Stripping do teacher bloat em checkpoints concluídos (`status=ok`, 122 dirs, sem risco de resume) | ~290 GB |
| Stripping do teacher bloat nos 404 dirs sem metadata (legado, provavelmente concluído — confirmar) | ~960 GB |
| Apagar diretórios inteiros de proj-types legados, se confirmado obsoletos (alternativa mais agressiva ao stripping acima) | até 1.3 TB |
| **Total potencial (stripping conservador, sem tocar nos 4 `status=error`)** | **~1.25 TB (≈ 89% do repo)** |

## 7. Comandos Recomendados

**Limpeza trivial (segura, execute quando quiser):**
```bash
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
```

**Stripping do teacher em checkpoints antigos (NÃO EXECUTAR sem revisão — reescreve arquivos de checkpoint):**
```python
# strip_teacher.py — remove pesos do teacher congelado de checkpoints legados,
# preservando exatamente a mesma lógica de FrozenTeacherCheckpointMixin.on_save_checkpoint.
# Rodar SÓ em runs com run_metadata["status"] == "ok" (ou confirmados como concluídos).
import sys, torch
path = sys.argv[1]
ckpt = torch.load(path, map_location="cpu", weights_only=False)
before = sum(v.numel() * v.element_size() for v in ckpt["state_dict"].values())
ckpt["state_dict"] = {k: v for k, v in ckpt["state_dict"].items()
                       if k.startswith(("student.", "proj_kd."))}
after = sum(v.numel() * v.element_size() for v in ckpt["state_dict"].values())
torch.save(ckpt, path)
print(f"{path}: {before/1e9:.2f} GB -> {after/1e9:.2f} GB")
```
```bash
# Exemplo de uso em um único diretório de run, só depois de confirmar status=ok:
python strip_teacher.py artifacts/distillation/<run>/checkpoints/last.ckpt
```

**Antes de rodar em massa nos 561 arquivos:** decidir com o usuário (i) se os proj-types legados ainda são usados nas comparações atuais (se não, apagar o diretório inteiro é mais simples que stripar) e (ii) confirmar que os 4 runs `status=error` não precisam de resume antes de tocar neles.
