"""Backbones autocontidos B0–B4 (Agente 3, missão SPiFiL progressivo).

Módulos (sem efeito colateral na importação; ``ga_search.modehb`` não é modificado):
* ``spec``          BackboneSpec v1 / GrowthSpec v1 (dataclasses, JSON canônico, hash);
* ``build``         spec -> nn.Module (ConvBlock + EmbeddingHead do repo), params e MACs/FLOPs medidos;
* ``budget``        faixas B0–B4, teto duro 5,5M, reparo da forma da cabeça (embed_dim >= 1280 mantido);
* ``morphisms``     widen (conv/BN/residual/dwsep/inception/ZCA/DBN) + head_widen, deepen_identity,
                    deepen_residual_zero, add_skip_zero, finalize -> GrowthSpec;
* ``preservation``  erro pai x filho em eval() (rel-L2, max|.|, cosseno) no mapa e no embedding;
* ``checkpoint``    arquivo único autocontido (spec + state_dict + TorchScript);
* ``family``        família de referência plana B0 -> B4 por morfismo exato.
"""
