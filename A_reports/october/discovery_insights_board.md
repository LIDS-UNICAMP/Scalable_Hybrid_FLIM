# Quadro de insights — discovery híbrido SPiFiL

Só se acrescenta texto aqui, nunca se edita ou apaga: `cat >> ... <<'EOF'` com um bloco por insight. Cabeçalho: `## <AAAA-MM-DD HH:MM> [agente] <título>`.
Cada bloco traz: o que rodou (YAML, work_dir), métricas (κ val por camada × dataset, κ teste quando houver), a conclusão e o que o próximo agente deve tentar ou evitar.
Agentes: **A0** = orquestrador (frente 1: baseline R1 e dilation R2), **A1** = frente da loss/objetivo, **A2** = frente da convolução (largura/kernel).

## 2026-10-30 00:55 [A0] R1 baseline, SVM em validação (flatten)
- R1 = layerwise 3, 5x5, 24/32/48, DISF/50, BCE, 200 épocas (growth_disc_r1.yaml).
- κ val: eggs L1 0.891 / L2 0.892 / L3 0.898 (2 splits). larvae split1: L1 0.796 / L2 0.822.
- A val_recon_loss trava em ~0.468 com ~60 épocas: a BCE para de ensinar cedo.
- Alvo (I-JEPA, teste): eggs 0.9397, larvae 0.9433, protozoan 0.8636. Faltam ~4 pontos em eggs.
