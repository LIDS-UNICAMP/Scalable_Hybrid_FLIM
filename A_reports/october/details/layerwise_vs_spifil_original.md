# layerwise contra o SPiFiL original: por que os filtros não batem (08/10/2026)

**Resumo em uma frase:** o layerwise usou **uma foto a mais** para montar os filtros (a da
impureza), e por isso sai com 9/27/45 filtros no eggs em vez dos 16/32/48 do SPiFiL original.

---

## 1. A diferença, em números

| | fotos usadas (uma por classe) | filtros |
|---|---|---|
| SPiFiL original, eggs | **8** (classes 1–8, sem impureza) | **16 / 32 / 48** |
| layerwise, eggs | **9** (classes 1–9, com a impureza, classe 9) | **9 / 27 / 45** |
| SPiFiL original, cistos | **6** (sem impureza) | **12 / 30 / 48** |
| layerwise, protozoan | **7** (com a impureza, classe 7) | **14 / 28 / 42** |
| larvae (os dois) | 2 (larva + impureza, **os dois usam a impureza**) | 16 / 32 (o layerwise para na L2) |

**A classe de impureza existe no dataset e é a maior de todas:** classe 9 do eggs (3.344 de
5.112 imagens, 65%), classe 7 do protozoan (5.716 de 9.568, 60%) e classe 2 do larvae (3.068 de
3.514, 87%). Ela faz parte da tarefa de classificação; a diferença é só se ela entra como foto
para recortar filtros. O original a deixa de fora no eggs e nos cistos, mas a usa no larvae,
onde sem ela sobraria uma classe só.

Onde isso foi conferido:
- **Imagens de exemplo do SPiFiL** (`SPiFiL/examples/data`): eggs tem 8, cistos 6 e larvae 2.
- **README do SPiFiL**: *"cysts builds 12/30/48 … (excluding impurities)"*.
- **Log do layerwise**: a classe 9 do eggs aparece como imagem-semente.

## 2. Explicação simples

O SPiFiL monta os filtros olhando **uma foto de cada tipo de parasita**, como um álbum com uma
página por parasita. Os filtros são recortes tirados dessas páginas.

- O **original** monta o álbum só com parasitas: 8 fotos no eggs.
- O **layerwise** pôs uma página a mais, a de **impureza** (sujeira, fundo): 9 fotos.

O SPiFiL divide os filtros **igualmente entre as fotos** e joga a sobra fora:

| | fotos | 16 filtros divididos | sobra no lixo | ficam |
|---|---|---|---|---|
| original | 8 | 2 para cada | 0 | **16** |
| layerwise | 9 | 1 para cada | 7 | **9** |

É como 16 balas para 8 crianças (2 para cada, nada sobra) contra 16 balas para 9 crianças
(1 para cada, 7 vão para o lixo).

**Consequências:**
- O layerwise ficou com **menos filtros** no eggs.
- Parte dos filtros dele foi recortada **da sujeira**, não do parasita.

## 3. De quem é cada parte

| | quem decide | original | layerwise |
|---|---|---|---|
| quantas fotos entram | o nosso `train_samples` | 8 (sem impureza) | **9 (com impureza)** |
| como os filtros são divididos | o SPiFiL (`allocator: uniform`) | igual, sobra fora | igual, sobra fora |

- **Dividir igual e jogar a sobra fora é o padrão do SPiFiL**, e o único modo que ele tem
  (`conf/allocator/` só tem `uniform.yaml`).
- **Receber 9 fotos é escolha do nosso código.** O `train_samples` pega uma foto de cada classe
  que existe no treino, e no nosso eggs as classes vão de 1 a 9. O código do SPiFiL não tem
  nenhuma regra de impureza: ele simplesmente nunca recebeu essa foto.

## 4. Outras diferenças para o original

1. **Impureza como semente.** É a diferença deste relatório.
2. **Superpixel:** o layerwise usa **SLIC com 100 regiões**; o artigo usa **DISF com 50**.

A pasta `data/to_mateus/model/` **não tem o SPiFiL original**. O que está lá é o **FLIM**: 24/32/48
filtros e kernel 5×5, enquanto o SPiFiL usa 3×3.

**Conclusão:** os números do layerwise continuam válidos como experimento, mas ele **não
reproduz exatamente** a configuração oficial do SPiFiL. Chamá-lo de "config oficial" não está
certo.

## 5. Onde está o código do layerwise

| o quê | arquivo |
|---|---|
| loop L1 → LN, filtros por camada (`LAYERWISE_SPECS`), checagem de congelamento | `experiments/ray/runners/growth.py:163-360` |
| escolha das fotos, **onde a impureza entra** | `flim/spifil.py:423` (`train_samples`) |
| corte da camada pelo SPiFiL; L1 direto da imagem | `flim/spifil.py:688` (`grow`), `:771` |
| congelar as antigas, decoder novo, SVM desligado | `methods/autoencoder/autoencoder_flim_module.py:258, 264, 326, 455, 527` |
| chaves `layerwise` e `evaluate.SVM` | `experiments/ray/schema.py:69, 322`; `experiments/ray/runners/train.py:97` |
| YAMLs | `experiments/autoencoder/growth_layerwise{3,5}.yaml`, `eval_svm_layerwise{3,5}.yaml` |

## 6. Como ficar fiel ao original

Não precisa mexer no SPiFiL. Basta o `train_samples` (`flim/spifil.py:423`) deixar a classe
de impureza de fora, de preferência controlado por uma chave no YAML, e **treinar a grade de
novo**. Para igualar também o superpixel, é preciso trocar SLIC/100 por DISF/50.
