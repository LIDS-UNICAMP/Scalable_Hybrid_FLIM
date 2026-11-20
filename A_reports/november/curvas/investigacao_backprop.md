# Por que a receita COM backprop fica pior (10/11)

**Resposta curta:** o braço com backprop nunca partiu da receita. O backprop quase não mexe nos pesos; o que está pior é a rede que o `grow` monta, e a leitura usada na comparação também é outra.

## Hipóteses

| # | veredito | evidência |
|---|---|---|
| H1 checkpoint por val/loss | **parcial** | Protozoan s1 p5: ckpt de L2 a L6 nas épocas 192–199, perto do fim, então a escolha pouco importa. L1 na época 32. Larvae p50 L6 na época 5 (0,209 contra 0,51 em p25) é o único caso onde a escolha pesa. Não há κ por época: `best_val_svm_kappa` = NaN em `L*/run_metadata.json`. |
| H2 a camada crescida não é a da receita | **confirmada (causa principal)** | ver lista abaixo. A L1 do checkpoint é idêntica ao `grow` (cos 1,000), então o κ de L1 (0,287 @1%, 0,511 @5%, 0,666 @50%) já mede a rede ANTES do backprop. A 50% ela fica bem abaixo dos 0,834/0,887 da receita. |
| H3 updates destrutivos | **refutada** | conv1 entre estágios: cos 1,0000 (L2 a L6). Só o weight decay encolhe a norma em 3–9%. A camada nova se move pouco: cos(grow, final) = 0,79 / 0,72 / 0,98 / 0,98 / 0,98 (L2 a L6). "lr 0,1" é um multiplicador (`old_layer_lr_scale`): o lr efetivo das camadas antigas é 5e-4×0,1 = 5e-5 (`methods/autoencoder/autoencoder_flim_module.py:380,1365-1376`). |
| H4 BCE premia copiar | **confirmada: não informa** | a val BCE fica plana em 0,6116–0,6158 de L1 a L6, igual a 1% e a 5%, enquanto o κ oscila entre 0,23 e 0,61. |
| H5 leitura diferente | **confirmada** | referência = L6 `cat` (+TTA 0,887), PCA-whitening. Backprop = GeM só da última camada (`gem_tta_pw`). A tabela mostra L2 > L6, e a leitura `cat` pegaria isso. Além disso, a comparação misturava 1%/5% com a referência a 50%. |

## Diferenças grow (growth runner) × receita (`spf_grade`, `experiments/autoencoder/disc_md_receita_tta.yaml`)
1. Amostra: o grow usa 1 imagem por classe com rótulo (`one_per_class` default, `flim/spifil.py:131,506`) e 252 canais (36 por classe × 7). A receita usa 500 imagens sem rótulo, FPS/fpsw e 256 canais.
2. Sem saliência (fpsw `mapa_w0.5_1` + pesos gmr) no grow.
3. Sem `entre_camadas` (whitening por canal com shrinkage entre camadas); o próprio YAML admite isso (`hyb_curva_receita.yaml:2`).
4. Normalização: a receita entrega direções de norma 1 no espaço z-score (`eval/spf_grade.py:21`). O grow usa kernels FLIM com o padrão do `architecture.json` (max_pool 3, stride 2).
5. As camadas k≥2 são crescidas sobre o checkpoint treinado da camada anterior, não sobre a receita.
6. L1 com `old_layers=0` cai em `freeze_encoder(except_last=True)` (`autoencoder_flim_module.py:533-535`): a conv1 treina com lr cheio, mas na prática não se moveu.

## Fine-tune correto da receita inteira
- Construir as 6 camadas com o próprio `spf_grade` (mesmas fontes, saliência, `entre_camadas`, 256, pools 2-2-2-2-1-1) e exportar os pesos COM o whitening dobrado na conv seguinte.
- Medir o κ da receita importada ANTES de qualquer passo (tem que dar ≈0,834), com a mesma leitura `cat`(+TTA).
- Fine-tune de ponta a ponta, com o mesmo lr nas 6 camadas.
- Escolher o checkpoint pelo κ de validação (sonda), nunca pela BCE. Guardar a época 0.
- Avaliar sempre com `cat`, à mesma porcentagem da referência.

## Resultado da receita inteira + autoencoder (10/11, jaci; parado pelo usuário)
Eggs, 50%, κ da sonda (a L0 é a receita pura) → época 10; a reconstrução de validação no mesmo intervalo:
- s1: 0,766 → 0,530 (recon 0,7240 → 0,6073)
- s2: 0,711 → 0,520 (recon 0,7248 → 0,6083)
- s3: 0,778 → 0,548 (recon 0,7257 → 0,6087)
O κ cai de forma monotônica desde a época 1, enquanto a reconstrução melhora: o autoencoder ensina a copiar a imagem e destrói a separação das classes que a receita tinha. O treino foi trocado por LeJEPA (SSL com SIGReg) inicializado com a mesma receita. Dados: `/dados/home/moliveira/exps/spifil_growth/hyb_receita_inteira/` (jaci; epoch_metrics.csv e ckpts L0/L6).
