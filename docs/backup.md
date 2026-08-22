# Backup do repositório para HD externo

Runbook para copiar `Scalable_Hybrid_FLIM` inteiro de **athena** para o HD externo
`/Volumes/Eduarda`, montado num Mac. Nada é excluído: os 46 G, incluindo `artifacts/`
e `logs/`.

## O que está sendo copiado

Levantamento feito em 2026-08-22, em `/dados/home/moliveira/Scalable_Hybrid_FLIM`:

| item | valor |
|---|---|
| tamanho total | 46 G |
| arquivos | 123.542 |
| `artifacts/` | 30 G |
| `logs/` | 16 G |
| `data/` | 669 M |
| arquivos de peso | 2.789 (2.387 `.ckpt`, 378 `.pth`, 24 `.pt`) |
| maior arquivo | 2,36 G (`artifacts/distillation/distillation_eggs_split2_pct100_2l_1x1_init_flim_256_1280/checkpoints/`) |
| symlinks | 4.124, dos quais 974 quebrados (apontam para `~/.cache/wandb/`, já limpo) |
| hardlinks | nenhum |

Os dois números que ditam o comando são os **symlinks** e o **maior arquivo**. Os 2,36 G
ficam abaixo do teto de 4 G do FAT32, então o tamanho de arquivo nunca é o problema. Os
symlinks quebrados, sim: eles decidem quais flags do rsync podem ser usadas.

## Os comandos — rodar **no Mac**, não em athena

`/Volumes` é caminho de macOS e não existe em athena. O HD está na ponta do Mac, então
quem puxa é o Mac.

**1. Rede.** Estando fora da Unicamp, suba a VPN antes. Teste:

```bash
ssh moliveira@143.106.60.198 "echo ok"
```

Sem o `ok` de volta, o resto não anda — é rede, não rsync.

**2. Espaço no destino.** Precisa de 46 G livres:

```bash
df -h /Volumes/Eduarda
```

**3. A cópia:**

```bash
rsync -a --partial --progress \
  moliveira@143.106.60.198:/dados/home/moliveira/Scalable_Hybrid_FLIM \
  /Volumes/Eduarda/
```

Cria `/Volumes/Eduarda/Scalable_Hybrid_FLIM/`. Se cair no meio, rode a mesma linha de
novo: `--partial` retoma de onde parou.

**4. Só se o passo 3 encher a tela de `symlink` / `Operation not supported`** — quer dizer
que o HD é ExFAT ou NTFS, que não sabem criar symlink. Aí empacote em arquivo único:

```bash
ssh moliveira@143.106.60.198 \
  "tar -cf - -C /dados/home/moliveira Scalable_Hybrid_FLIM" \
  > /Volumes/Eduarda/Scalable_Hybrid_FLIM.tar
```

O tar guarda symlink, permissão e data dentro do próprio arquivo, então o formato do HD
deixa de importar. Em troca, não dá para retomar: se cair, recomeça.

Para saber o formato antes de tentar, sem esperar o erro:

```bash
diskutil info /Volumes/Eduarda | grep -i "File System"
```

**5. Conferência final:**

```bash
find /Volumes/Eduarda/Scalable_Hybrid_FLIM | wc -l   # tem que dar 123542
du -sh /Volumes/Eduarda/Scalable_Hybrid_FLIM          # tem que dar 46G
```

## Por que estas flags

**`-a`** — recursivo preservando permissão, dono, data e **symlink como symlink**. É o que
faz os 974 links quebrados serem copiados sem erro: o rsync copia o link, não o alvo.

**Nunca `-L`.** Essa flag manda seguir o link e copiar o arquivo apontado — os 974
quebrados viram 974 erros, e os bons passariam a puxar arquivo de fora do repositório.

**`--partial`** — guarda o pedaço transferido se a conexão cair. Com 46 G isso é a
diferença entre retomar e recomeçar.

**`--progress`, e não `--info=progress2`** — o rsync que a Apple distribui é antigo e não
conhece `--info=progress2`. Se `rsync --version` no Mac disser 3.x (via Homebrew), pode
trocar e ganhar uma barra do total em vez de uma por arquivo.

**Sem `-H`** — não há hardlink no repositório, então preservá-los não faz diferença.

## Duas pegadinhas

**A barra no fim da origem.** `.../Scalable_Hybrid_FLIM` cria a pasta dentro do destino;
`.../Scalable_Hybrid_FLIM/` despeja o conteúdo solto em `/Volumes/Eduarda/`. Os comandos
acima usam a forma sem barra.

**O Mac dormindo no meio.** São 46 G por rede — mais de uma hora numa conexão de 100 Mbit.
Prefixe com `caffeinate -i` para o laptop não suspender:

```bash
caffeinate -i rsync -a --partial --progress \
  moliveira@143.106.60.198:/dados/home/moliveira/Scalable_Hybrid_FLIM \
  /Volumes/Eduarda/
```

## Variante enxuta (não é o que foi pedido)

Fica registrado por ser útil noutra ocasião: `artifacts/` e `logs/` são reproduzíveis por
retreino e somam 46 dos 46 G. Sem eles o backup cai para menos de 1 G:

```bash
rsync -a --partial --progress \
  --exclude=artifacts/ --exclude=logs/ --exclude=wandb/ \
  moliveira@143.106.60.198:/dados/home/moliveira/Scalable_Hybrid_FLIM \
  /Volumes/Eduarda/
```

## Nota sobre `scalable_FLIM_self_supervised`

Não precisa entrar no backup. Comparação de 2026-08-22: dos 422 arquivos daquele
repositório, 421 já existem aqui com o mesmo caminho relativo — o único ausente é
`README_UV.md`. Os 51 arquivos que diferem têm aqui a versão mais nova (código que lá
estava duplicado foi centralizado em `src/evaluate/constants.py` e `src/utils/evaluate.py`).
Ele não tem nenhum arquivo de peso, e o `data/to_mateus` dele é um symlink que aponta
para dentro deste repositório.
