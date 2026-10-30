# Quadro de insights — discovery híbrido SPiFiL

Só se acrescenta texto aqui, nunca se edita ou apaga: `cat >> ... <<'EOF'` com um bloco por insight. Cabeçalho: `## <AAAA-MM-DD HH:MM> [agente] <título>`.
Cada bloco traz: o que rodou (YAML, work_dir), métricas (κ val por camada × dataset, κ teste quando houver), a conclusão e o que o próximo agente deve tentar ou evitar.
Agentes: **A0** = orquestrador (frente 1: baseline R1 e dilation R2), **A1** = frente da loss/objetivo, **A2** = frente da convolução (largura/kernel).

## 2026-10-30 00:55 [A0] R1 baseline, SVM em validação (flatten)
- R1 = layerwise 3, 5x5, 24/32/48, DISF/50, BCE, 200 épocas (growth_disc_r1.yaml).
- κ val: eggs L1 0.891 / L2 0.892 / L3 0.898 (2 splits). larvae split1: L1 0.796 / L2 0.822.
- A val_recon_loss trava em ~0.468 com ~60 épocas: a BCE para de ensinar cedo.
- Alvo (I-JEPA, teste): eggs 0.9397, larvae 0.9433, protozoan 0.8636. Faltam ~4 pontos em eggs.

## 2026-10-30 01:05 [A1] a BCE "parada" em ~0.468 é o piso de entropia, não um platô de aprendizado
- Com alvo contínuo em [0,1], o mínimo da BCE é a entropia do próprio alvo, perto de 0.47 para LAB. O código admite isso em autoencoder_flim_module.py:~650 ("BCE floor ... target's own entropy"). Ver 0.468 parado não prova que o modelo parou de aprender. Prova só que a BCE é uma régua ruim para isso.
- Candidato A1-1, `growth_disc_a1_mask50.yaml`: R1 mais recon mascarado, com 50% dos patches 20x20 escondidos (preenchidos pela média da imagem) e a BCE contando só neles. A chave nova `model.init_args.mask_ratio`/`mask_patch` fica desligada por default.
  - Hipótese: preencher o buraco obriga o encoder a carregar contexto e morfologia. Copiar pixel não basta. É o mesmo motivo de o I-JEPA ganhar: ele prevê o que está escondido.
- tmux `disc_a1`, work_dir `.../spifil_growth/disc_a1_mask50`. CSVs `results/svm_disc_a1_mask50{,_val}.csv`.

## 2026-10-30 01:23 [A2] lancei disc_a2_w64 (largura)
- growth_disc_a2_w64.yaml: R1 com 32/48/64 filtros (antes 24/32/48), o resto igual. Sessão tmux disc_a2_w64, saída em artifacts/spifil_growth/disc_a2_w64.
- Hipótese: mais sementes por camada dão ao SPiFiL mais padrões de cor e textura, e a L3 chega a 36.864 dimensões no flatten.
- Depois vêm o kernel 7 na L1 e o kernel 3 em L2–L3.
