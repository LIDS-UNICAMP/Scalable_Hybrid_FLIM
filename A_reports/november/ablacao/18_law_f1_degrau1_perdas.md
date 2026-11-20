# law_f1 degrau 1 (qual perda SSL) — Fase com backprop

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `disc_md_law_f1_{barlow,corinfo,recon,strong,urjepa,vicreg,vicregl,weak}_*`; SVM `disc_md_law_eval_f1`
- **Alternativa:** UR-JEPA, Strong/Weak-SIGReg, Barlow, VICReg, VICRegL, CorInfoMax, só reconstrução; λ 1e-2 a 1e-4
- **O que testa:** perda SSL para a lei (fase com backprop, pausada em 07/11 a pedido do usuário)
- **Como:** B0, 200 px, 50 superpixels, split 1, semente 42, 15 épocas por estágio; `gem_tta_pw` L4
- **Bases:** flowers, heggs, hlarv
- **Resultados:** Δκ médio contra só reconstrução: UR-JEPA 0,01 +0,024; Strong-SIGReg 0,001 +0,018; Barlow 0,001 +0,016; VICRegL 0,0001 +0,015; VICReg 0,01 +0,012; só reconstrução κ médio 0,796
- **Estado:** concluído
- **Decisão:** os 5 primeiros vão ao degrau 2; ordem do topo dentro do ruído (~0,015)

## Fontes
- [`results_log.md`](../multi_dataset/results_log.md)
