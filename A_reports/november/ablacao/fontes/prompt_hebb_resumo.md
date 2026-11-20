# Prompt do usuário, 07/11 23:20 UTC: SPiFiL + SoftHebb com seleção adaptativa de kernels

Ordem: pôr na fila, aproveitar o que já existe, mandar tudo possível para o Jaci; o que não couber, no Athena. Sem rótulos na construção, atualização, avaliação e substituição dos kernels. Sem backprop. Auditar antes; não duplicar. Mesmas métricas, bases, splits, sementes, classificador e diretórios.

Fila 1 (efeito básico): E00 FPS fixo; E01 SoftHebb do aleatório; E02 FPS + SoftHebb (guardar antes e depois); E03 FPS + SoftHebb limitado (intensidade e duração; restrição de distância ao kernel inicial; livre × restrito).
Fila 2 (homeostase): E04 competitiva (frequência de vitória, viés adaptativo, sem igualar frequências); E05 seletiva (só kernels persistentemente inativos; proteger raros estáveis; desligar ao recuperar).
Fila 3 (diagnóstico e proteção): E06 cobertura marginal (remoção virtual); E07 estabilidade sob perturbação (alinhar mapas; poucos dados = inconclusivo); E08 diversidade funcional (assinaturas de ativação, grupos redundantes, representantes); E09 suporte de padrões raros (patches parecidos em várias imagens; frequência não basta).
Fila 4 (robustez e recuperação): E10 FPS robusto a outliers + SoftHebb; E11 recuperação de inativos não protegidos; E12 substituição de redundantes com FPS em regiões mal cobertas (orçamento constante); E13 substituição com validação sem rótulo (aceita ou restaura).
Fila 5 (combinações): E14 FPS + SoftHebb + proteção; E15 + substituição adaptativa; E16 proteção + substituição; E17 tudo junto.
Fila 6 (políticas): E18 substituição conservadora × agressiva; E19 por camada (1, intermediárias, profundas, todas; patches do espaço de ativação de cada camada); E20 congelamento progressivo.

Regras: não descartar só por baixa frequência; redundância por resposta, não só por peso; não forçar frequências iguais; orçamento constante; só treino para construir e calibrar; validação sem rótulo para aceitar substituição; checkpoints antes e depois. Isoladas antes das combinações.
