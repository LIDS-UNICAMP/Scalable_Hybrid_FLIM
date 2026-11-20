# SPiFiL sem backprop: estado em 08/11, 12:40 UTC

Só validação. Régua: melhor aleatório de qualquer configuração. 5 famílias: ovos, larvas, protozoários, aid + remote_sensing, flowers (coconut_trees e food101 entrando).

## Receita única
`fpsw@mapa_w0.5_1` (`fps` + peso da saliência `gmr`, corte 0,5), 6 camadas concatenadas, 256 filtros, whitening entre camadas, 50 superpixels DISF, GeM + PCA-whitening + SVM. Pior arrependimento −0,029 entre as famílias; ganho sobre o aleatório: ovos +0,199, larvas +0,290, protozoários +0,210, aid + remote_sensing +0,028, flowers +0,067. Regra do usuário: sem objeto definido (aid, remote_sensing), desligar a saliência (`fps`).

## Melhor SPiFiL por base (L6) e DINOv2
helminth_larvae 0,972 / 0,994; helminth_eggs 0,958 / 0,987; protozoan_cysts 0,921 / 0,966; Protozoan_subgroup 0,906 / 0,945; Eggs_subgroup 0,893 / 0,989; Larvae_subgroup 0,885 / 0,981; Eggs 0,866 / 0,980; Protozoan 0,863 / 0,939; Larvae 0,855 / 0,968; aid 0,775 / 0,970; flowers 0,690 / 0,973; remote_sensing 0,508 / 0,708.

## Funciona
Whitening entre camadas (maior ganho); saliência como peso ou foco nas bases com objeto; `fps`; 6 camadas concatenadas; 256 filtros (exceto imagens pequenas, ver coconut_trees); leitura multiescala (pequeno).

## Não funciona
Normalizações novas (Z-score do patch, global, MAD); agrupamento; ZCA ou PCA nos filtros; multiescala e DoG; densidade SIFT e fusão Top-K como mapa; diversidade pelas respostas; GAP, GMP e SPP; mais de 256 filtros; SoftHebb e homeostase.

## Em andamento
Imagens por classe (8 bases); polaridade; `r4` → `r5`; food101 e coconut_trees em todos os experimentos; relação tamanho da imagem × arquitetura.

## Próximos passos
Regra com e sem saliência (`if`/`else` do usuário); confirmar a receita com mais sorteios; backprop a partir da receita.
