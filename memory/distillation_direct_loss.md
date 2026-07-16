# Modo `direct` — Destilação FLIM CNN ← I-JEPA

Treina a FLIM CNN (pesos trunc_normal) para produzir embeddings próximos do I-JEPA via MSE no espaço projetado. Sem labels, puramente self-supervised.

## Fluxo

```
x  [B, 3, 200, 200]
│
├──────────────────────────────────────┐
▼                                      ▼
FLIM CNN encoder (com grad)     resize bilinear → [B, 3, 224, 224]
│                                      │
[B, 48, 24, 24]               I-JEPA ViT-H/14 (frozen, sem grad)
│                                      │
ConvProjectionHead (1×1 convs)  mean-pool patches
│                                      │
student_proj [B, 1280]          teacher_emb [B, 1280]
│                                      │
└────────────── MSE ───────────────────┘
               loss = mse(student_proj, teacher_emb)
               backprop → atualiza FLIM encoder + proj_kd
```

## Componentes

| Componente | Arquivo |
|---|---|
| `MSEDistillationLoss` | `src/models/distillation.py` |
| `ConvDistillationProjectionHead` | `src/models/distillation.py` |
| `FrozenTeacher` / `IJEPAEncoder` | `src/models/distillation.py` / `ijepa_encoder.py` |
| `DistillationConvModule` | `src/modules/distillation_conv_module.py` |
| Launcher | `scripts/distillation_conv_ray.py` |

## Parâmetros padrão

- `encoder_init = trunc_normal`
- Classes por dataset: eggs=9, larvae=2, protozoan=7
- Grid: 3 datasets × 3 splits × 6 pcts = 54 runs
- `--max-concurrent-per-gpu 1` obrigatório (I-JEPA usa ~5 GB por processo)
