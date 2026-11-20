# Prompt guardado: lei de escala do backbone (rodar no fim da descoberta)

Pedido do usuário em 2026-11-02. Rodar só quando a descoberta terminar, isto é, quando eggs/larvae/protozoan empatarem com o I-JEPA ou quando a grade parar de subir. Até lá, este arquivo só guarda o prompt.

Pergunta: achamos um backbone que empata com o I-JEPA em larvae e passa dele em protozoan. Se a gente tirar metade dele (e um quarto, um oitavo...), o κ cai seguindo uma lei de escala?

---

## Prompt (colar como está)

> Contexto: worktree `/dados/home/moliveira/Scalable_Hybrid_FLIM_discovery`, branch `discovery_hybrid_spifil`.
> - A receita do backbone final está em `A_reports/october/discovery_recipe.md`.
> - O ranking está em `A_reports/october/discovery_ranking_test.md`.
> - O log das decisões está em `A_reports/october/discovery_hybrid_spifil_decisions.md`.
> - Use a família de referência que estiver no topo do ranking pela **val** na hora de rodar; hoje é `disc_bp01w64c1280` / `disc_g20_base_e60`.
> - CNN de 4 camadas 32/48/64/1280, kernels 5/5/5/1, pools 2/2/2/1.
> - SPiFiL inicializa cada camada; a 1x1 é completada por `fill_random_last`.
> - As camadas antigas treinam com backprop ×0.1, e o fundo pesa 0.1.
> - Leitura: GeM p=3 na última camada, depois SVM.
> - I-JEPA, teste 50%: eggs 0.9397, larvae 0.9433, protozoan 0.8636 (`/dados/home/moliveira/Scalable_Hybrid_FLIM/results/ijepa_svm_aggregated.csv`).
>
> Objetivo: medir uma **lei de escala do tamanho do modelo** para este problema. Encolha o backbone e veja se o κ cai de forma previsível com o número de parâmetros N.
>
> Desenho (uma mudança por família, sempre com SPiFiL inicializando):
> 1. **Eixo largura (o principal):** multiplique todas as larguras das convs por m ∈ {1/8, 1/4, 1/2, 1, 2}, mantendo a profundidade e a 1x1 escalada junto.
>    - Exemplos: m=1/2 dá 16/24/32/640; m=2 dá 64/96/128/2560. A 2560 passa de 2000-d, então para a leitura use 1x1 ≤1999 ou GeM da L3; não use flatten.
>    - Os pontos m ≤ 1 são a pergunta do usuário ("tirar metade").
> 2. **Eixo largura da 1x1** (separado, com o resto fixo): 1x1 ∈ {160, 320, 640, 1280, 1920}.
> 3. **Eixo profundidade** (separado): 2, 3, 4, 5, 6 camadas SPiFiL antes da 1x1, reaproveitando as famílias `e_d5`/`e_d6` se já tiverem saído.
> 4. Os três datasets, splits 1/2/3, 50% dos dados. Use o orçamento de épocas que for o padrão no fim da descoberta (60 se o `base_e60` passou).
>
> Medidas por ponto:
> - N = parâmetros do encoder, de `L*_grow/architecture.json`;
> - FLOPs por imagem;
> - κ val e κ teste (gem na última camada);
> - **fração SPiFiL** na 1x1: filtros SPiFiL ÷ filtros totais. Eggs tem 400 sementes, então uma 1x1 de 320 é 100% SPiFiL e uma de 1280 é 31%. Isso confunde a lei e tem que ser reportado e controlado.
>
> Ajuste da lei, só com a val; o teste é só para reportar:
> - erro e = 1 − κ;
> - ajuste e(N) = E + A·N^(−α) por dataset (scipy `curve_fit`), com IC de α por bootstrap sobre os splits;
> - validação da lei: ajuste sem o maior ponto, extrapole para ele e reporte o erro da extrapolação;
> - lei conjunta com a curva de dados que já existe (família `disc_bp01w64c1280_pcts`, 1–100%): e(N, D) = E + A/N^α + B/D^β (forma Chinchilla). A pergunta final é se compensa mais crescer o modelo ou os dados.
>
> Entregáveis:
> - `A_reports/october/discovery_scaling_law.md`: tabela N × κ por dataset, os ajustes (E, A, α com IC), o gráfico log-log e uma frase por dataset em nível criança ("dobrar o modelo tira X% do erro");
> - onde a lei prevê que o modelo empata com o I-JEPA (N*) e se isso cabe em ≤5M params.
>
> Regras (todas valem):
> - experimento só via YAML em `experiments/autoencoder/`, rodado com `python -m experiments.ray.launch`, validado por `experiments/ray/schema.py`;
> - nunca `.sh`, nunca flag de argparse;
> - teto de 20 GB por GPU (soma dos processos do moliveira); o que não couber vai para a fila;
> - SVM só depois do treino, um por vez sob flock, `nice -n 10`;
> - sem flatten, e a leitura sempre abaixo de 2000-d;
> - teste nunca escolhe nada;
> - não sobrescrever resultados antigos;
> - registrar cada família no log de decisões com a hipótese SPiFiL;
> - commits sem trailer nem assinatura;
> - múltiplos agentes: um gera os YAMLs, um vigia as GPUs, um ajusta a lei.

---

## Por que isso importa
- Se o erro cai como N^(−α) com α estável entre datasets, a receita vira uma regra de projeto: dá para escolher o tamanho do modelo para um dataset novo sem refazer a grade.
- O cruzamento com a curva de dados (pcts) diz se, neste problema, vale mais crescer o modelo ou crescer os dados.
