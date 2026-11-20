# Resumo do dia: 10/11/2026

Tudo medido em validação (κ de Cohen), 50% dos dados e média de 3 splits, salvo quando indicado. A régua aleatória tem o **mesmo tamanho** do SPiFiL: random_z, 6×256, 8,2 mi parâmetros.

## 1. TTA na receita sem backprop
O TTA classifica a média de 8 versões da imagem, 4 rotações de 90° com e sem espelhamento. Relatório: `multi_dataset/tta_receita.md`.

| base | receita | receita + TTA | aleatório (régua) | I-JEPA | DINOv2 |
|---|---|---|---|---|---|
| Eggs | 0,866 | **0,919** | 0,560 | 0,970 | 0,980 |
| Protozoan | 0,834 | **0,887** | 0,485 | 0,921 | 0,939 |
| Larvae | 0,855 | **0,878** | 0,325 | 0,850 | 0,968 |
| aid | 0,771 | **0,844** | 0,635 | — | 0,970 |
| remote_sensing | 0,492 | 0,539 | 0,487 | — | 0,708 |
| flowers | 0,655 | 0,660 | 0,549 | — | 0,973 |

- O TTA ajuda onde a orientação não importa: parasitos e cenas aéreas. Na flowers não muda nada.
- Com TTA, o SPiFiL sem saliência alcança a receita nos parasitos.
- Na Larvae, o SPiFiL com TTA passa o I-JEPA.

## 2. Atenção sem parâmetros
Arquivos: `atencao/auditoria.md` e `atencao/triagem.md`.
- **Triagem:** 5 mecanismos (sa, espacial, SimAM, non-local, multiescala), aplicados na L5 e no mapa final. Ganhos de +0,01 a +0,03 num split.
- **Confirmação em 3 splits:** −0,015 a +0,007. Os ganhos da triagem eram ruído, e a atenção não entra na receita.
- **Combinação:** no remote_sensing, 320 px + espacial@L5 + TTA dá **+0,079** (3 de 3 splits). No aid, o TTA sozinho é o melhor (+0,073). Na flowers, nada.

## 3. Pontos de "register" do DINOv2
Figura: `analise_textura/dino_registers_pontos.png`.
- Os pontos soltos no mapa do DINOv2 são o artefato de Darcet et al. Sem registers, 1 a 4% dos tokens têm norma acima de 150; com registers, ~0.
- Recortar os filtros nesses pontos não ajuda: empata com o sorteio e, na flowers, perde 0,08 para a receita, porque os pontos caem no fundo liso.

## 4. SoftHebb, CLAPP e SFA na Larvae
Arquivo: `analise_textura/softhebb_clapp_sfa_larvae.md`.

| técnica | κ antes → depois | energia na larva |
|---|---|---|
| SoftHebb (rede de 3 camadas) | 0,679 → 0,732 | sobe 2 a 5 pontos |
| CLAPP (L4/L5) | 0,771 → 0,771 | igual |
| SFA (L4/L5) | 0,848 → 0,818 | cai ~5 pontos (vai para o fundo) |

## 5. Análises de filtros e mapas
Arquivos em `analise_textura/` e `analise_textura/kernels_classe/`.
- **Mapas de ativação L1:** a ativação dentro do objeto é 1,5× a 2,9× a área dele (Eggs, Protozoan, Larvae). As impurezas também acendem.
- **Nota Fisher:** não prevê a utilidade do filtro (ρ = −0,006). Escolher pelo Fisher dá mais filtros "fiéis", o que não é o mesmo que κ maior.
- **I-JEPA, DINOv2 (com e sem registers) e SPiFiL:** só o SPiFiL e o DINOv2 com registers separam o objeto do fundo no PCA. A norma do I-JEPA não serve como mapa.

## 6. Backprop
- **Curva antiga (layerwise):** pontos completos do Protozoan com κ 0,241 (1%) e 0,499 (5%). A melhor camada era sempre a L2.
- **Investigação** (`curvas/investigacao_backprop.md`):
  - a rede **nunca partiu da receita**: o grow usa 1 imagem por classe, com rótulo, sem saliência e sem whitening;
  - o lr real era 5e-5;
  - a perda de reconstrução fica parada;
  - a leitura era outra.
- **A curva antiga foi parada** no athena e no jaci.
- **Experimento novo, a receita inteira + autoencoder:** a receita SPiFiL é montada inteira e depois só a reconstrução é treinada com backprop (decoder descartado), e a avaliação é feita com SVM.
  - Antes de treinar, o κ confere com a receita: 0,825 em Protozoan s1.
  - **Zero rótulos**: nem no treino, nem na escolha do checkpoint (última época; o κ por época fica só para comparar).
  - No jaci, Eggs e Larvae estão treinando e Protozoan começa ~03:00. Depois vêm aid, remote_sensing, flowers e ham10000.

## 7. Rodando ao fim do dia
- **jaci:** receita inteira + autoencoder. A fila de explorações (e07, GELU, gaussiana, porclasse, largura, leis, 448 px no aid) e a grade do I-JEPA estão pausadas e retomam depois.
- **athena:**
  - HAM10000 com receita + TTA, de 1% a 100%;
  - 320 px + atenção + saliência em flowers, ham10000 e remote_sensing;
  - mecanismos de relação global (modo de construção, Fourier, covariância, contexto global, dilatação).

## 8. Regras novas do usuário
- O aleatório da régua tem sempre o mesmo tamanho do SPiFiL.
- Backprop nunca usa rótulos, inclusive na escolha do checkpoint.
- Tudo roda no jaci; o athena só recebe o que não couber lá.
