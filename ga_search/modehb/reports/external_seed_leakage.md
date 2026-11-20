# Vazamento transdutivo dos encoders externos "SPiFiL pequeno + LeJEPA" (seeds do MO-DEHB)

Data: 2026-11-15, host jaci. Somente leitura, exceto os symlinks em `artifacts/external_ssl_pequeno/`.
Script (temporário): `/tmp/claude-1014/-dados-home-moliveira-Scalable-Hybrid-FLIM/05a95f35-17a2-4976-a2bd-9c4d1e4ff9eb/scratchpad/leak_check.py`.

## Como foi verificado

- **Split deles** (config.yaml + código em `Scalable_Hybrid_FLIM_discovery`): `core.data.ParasiteDataModule(split=1, percentage=100)`
  -> `config.get_single_parasite_paths` -> `organized/<Base>/splits_incremental/split1/data_descriptor_perc100.json`;
  imagens em `core/constants.ORGANIZED`: `raw/orig` (Eggs/Larvae/Protozoan), `orig` (demais).
  Para Eggs/Larvae/Protozoan/aid/flowers/coconut/remote_sensing esse arquivo é **byte-idêntico** a `splits/split1.json`;
  no ham10000 ele tem os mesmos conjuntos que `split1.json` (ordem diferente), com `validation == train` (5007).
- **O que eles usaram**: o SSL (L6) treina no `train`. O checkpoint `best_kappa.ckpt` é escolhido pela menor `probe/val_ssl_loss`
  (sem rótulo) no `validation` (`monitor_spec`, `configure_callbacks`). A sonda por época lê só train+val e só é gravada.
  O L0 (FPS SPiFiL, `eval/spf_grade` + `flim/spifil.py`) usa só imagens de `train`. Nenhum desses passos toca o `test`.
- **Cache de imagens** (`/dados/home/moliveira/exps/image_cache/<Base>/200_nocrop`): a lista de arquivos é igual à de `raw/orig`
  (`orig` no ham10000) e não à de `pre/orig`, então os nomes do split realmente apontam para as imagens raw.
- **Identidade das imagens**: o mesmo nome de arquivo no mesmo diretório (bases sem pre/raw e larvae_raw); `raw_modehb/orig/<rótulo>_<raw>`
  passa por `realpath` até chegar em `raw/orig/<raw>` (eggs_raw/protozoan_raw); para pre, `splits/split1_modehb_mapping.csv` (new_name_pre -> old_name_raw),
  cuja coluna split bate 100% com `split1_modehb.json`. Os sha256 dos nossos splits batem com os `split_sha256` dos YAMLs.
- **Rótulos**: o conjunto de classes deles é igual ao nosso em todas as bases. Nas bases de parasitos, o rótulo de cada imagem
  (manifest `class_id`) bate com o nosso (prefixo pre / prefixo raw_modehb), com 0 divergências.
- **Duplicatas byte-a-byte** entre o train/val deles e o nosso test (pré-filtro por tamanho + md5): só 1, em flowers (ver notas).

## Tabela

| base | encoder (L0 / lejepa) | split deles | nosso split | train∩test | val∩test | train∩val | veredito |
|---|---|---|---|---|---|---|---|
| remote_sensing | artifacts/external_ssl_pequeno/remote_sensing_split1_pct100 (cópia athena, fps) | perc100 == split1.json | splits/split1.json | 0 | 0 | 0 | USE |
| flowers | artifacts/.../flowers_split1_pct100 (athena; L0 md5 == jaci fpsw@mapa_w0.5_1) | perc100 == split1.json | splits/split1.json | 0 | 0 | 0 | USE |
| coconut_trees | artifacts/.../coconut_trees_split1_pct100 (athena, fps) | perc100 == split1.json | splits/split1.json | 0 | 0 | 0 | USE |
| aid | artifacts/.../aid_split1_pct100 (L0 md5 == jaci fps) | perc100 == split1.json | splits/split1.json | 0 | 0 | 0 | USE só L0: `lejepa/checkpoints/` vazio, não existe aid pct100 L6 no jaci |
| ham10000 | encoders/ssl_pequeno/ham10000/fps + spifil_growth/.../ham10000_split1_pct100/L6 | perc100 (val == train, 5007) | splits/split1_modehb.json | 0 | 0 | 878 (todo o nosso val) | USE (ressalva) |
| larvae | encoders/.../Larvae/fpsw@mapa_w0.5_1 + .../Larvae_split1_pct100/L6 | perc100 == split1.json (raw) | splits/split1_modehb.json (pre) | 0 | 0 | 0 | USE |
| eggs | encoders/.../Eggs/fpsw@mapa_w0.5_1 + .../Eggs_split1_pct100/L6 | perc100 == split1.json (raw) | splits/split1_modehb.json (pre) | 0 | 0 | 0 | USE |
| protozoan | encoders/.../Protozoan/fpsw@mapa_w0.5_1 + .../Protozoan_split1_pct100/L6 | perc100 == split1.json (raw) | splits/split1_modehb.json (pre) | 0 | 0 | 0 | USE |
| larvae_raw | igual a larvae | perc100 == split1.json (raw) | splits/split1_modehb_raw.json | 0 | 0 | 0 | USE |
| eggs_raw | igual a eggs | perc100 == split1.json (raw) | splits/split1_modehb_raw.json | 0 | 0 | 0 | USE |
| protozoan_raw | igual a protozoan | perc100 == split1.json (raw) | splits/split1_modehb_raw.json | 0 | 0 | 0 | USE |

(val = validation deles; train∩val = train deles ∩ o nosso validation. Fora o ham10000, o val deles é igual ao nosso val.)
O split1_modehb(_raw) é a **mesma partição** do split1.json raw deles: os conjuntos train/val/test são iguais depois do mapeamento.

## Notas

- **ham10000**: train(deles) = o nosso train + o nosso val (o pool antigo de 5007), e a seleção do ckpt usa val == train. Então o
  encoder viu em SSL, sem rótulo, todas as 878 imagens do nosso val: isso dá ao seed uma vantagem otimista no objetivo de val da busca.
  No nível de imagem o test fica limpo. No nível de lesão, 1312 imagens de test compartilham `lesion_id` com o train deles,
  contra 1091 com o nosso train. É herança do test fixo (`known_issues: test_shares_lesions_with_train`).
- **protozoan/protozoan_raw**: eles têm 26162/5607/5607 e nós 26159/5607/5606. A diferença são as 4 imagens que não existem em pre/orig
  (3 de train e 1 de test, do lado deles e do nosso), então não há cruzamento.
- **pre vs raw**: o encoder externo de eggs/larvae/protozoan foi treinado em `raw/orig` (fundo original). Nas bases pre (fundo preto) não há
  vazamento, mas há troca de domínio. O mesmo encoder serve tanto à base pre quanto à base raw.
- **flowers**: `train/000005_00000207.jpg` é byte-idêntica a `test/000003_00000104.jpg` (e com classe diferente). É uma duplicata do próprio
  dataset, que já está no nosso split1.json e não foi introduzida pelo encoder.
- **Escolha da arquitetura** do L0 ("ESCOLHA PROVISORIA: kappa ... só no split 1" nos `hyb_ssl_encoder_*_jaci.yaml`): ela vem do kappa de validação dos
  runs restr5m e não do test.
- Bases athena (remote_sensing, coconut_trees): o config.yaml copiado diz `parasite_name=<base>, split 1, percentage 100`. O conteúdo do split
  foi conferido com a cópia do jaci, porque o arquivo do athena não foi lido.

## Symlinks criados (2026-11-15)

`artifacts/external_ssl_pequeno/<base>_split1_pct100/{L0,lejepa}` para ham10000, eggs, eggs_raw, larvae, larvae_raw, protozoan, protozoan_raw:
- `L0` -> `/dados/home/moliveira/exps/encoders/ssl_pequeno/<Base>/<fps|fpsw@mapa_w0.5_1>/<Base>_split1_pct100`, conferido como igual ao `flim_weights_path` do L6
- `lejepa` -> `/dados/home/moliveira/exps/spifil_growth/ssl_pequeno_lejepa/<Base>_split1_pct100/L6` (o `checkpoints/best_kappa.ckpt` resolve)
