# Hybrid SPiFiL — κ de TESTE com o SVM ORIGINAL do repositório vs modelos grandes
_Atualizado: 19/11 10:07. SVM original = core.svm.fit_svm (SVC linear, C=100, sem scaler) no embedding do modelo, mesmo protocolo dos FMs. Coluna cov_head = leitura do fitness (C=100). Seleção na validação; teste só reportado._

## aid

| modelo | κ teste SVM original | κ teste cov_head | κ val (fitness) | fid | params | estrato | busca |
|---|---|---|---|---|---|---|---|
| **AIMv2** (congelado) | 0.979 | — | — | — | — | FM | baseline |
| **DINOv2** (congelado) | 0.970 | — | — | — | — | FM | baseline |
| **I-JEPA** (congelado) | 0.916 | — | — | — | — | FM | baseline |
| **ResNet-50** (congelado) | 0.898 | — | — | — | — | FM | baseline |
| **ConvNeXtV2-pico** (congelado) | 0.888 | — | — | — | — | FM | baseline |
| c00101 | 0.731 | 0.837 | 0.805 | 45 | 198k | tiny | hybrid C100_kw1 (modehb_hybrid) |
| c00025 | 0.719 | 0.849 | 0.819 | 200 | 213k | tiny | hybrid C100_kw1 (modehb_hybrid) |
| c00025 | 0.719 | 0.849 | 0.819 | 5 | 213k | tiny | hybrid C100_kw1 (modehb_hybrid) |
| c00025 | 0.710 | 0.846 | 0.817 | 100 | 213k | tiny | hybrid C100_kw1 (modehb_hybrid) |
| c00025 | 0.710 | 0.848 | 0.820 | 45 | 213k | tiny | hybrid C100_kw1 (modehb_hybrid) |
| c00020 | 0.704 | 0.867 | 0.840 | 15 | 2137k | huge | hybrid C100_kw1 (modehb_hybrid) |
| c00025 | 0.701 | 0.851 | 0.820 | 15 | 213k | tiny | hybrid C100_kw1 (modehb_hybrid) |
| c00099 | 0.684 | nan | 0.811 | 45 | 1082k | medium | hybrid C100_kw1 (modehb_hybrid) |
| c00091 | 0.678 | 0.858 | 0.839 | 45 | 740k | small | hybrid C100_kw1 (modehb_hybrid) |
| c00034 | 0.678 | 0.856 | 0.821 | 45 | 308k | small | hybrid C100_kw1 (modehb_hybrid) |
| c00001 | 0.675 | 0.685 | 0.668 | 45 | 23k | tiny | hybrid C0.1_kw0.75 (modehb_hybrid_C01_2049) |
| c00001 | 0.675 | 0.685 | 0.663 | 100 | 23k | tiny | hybrid C100_kw1 (modehb_hybrid) |

## coconut_trees

| modelo | κ teste SVM original | κ teste cov_head | κ val (fitness) | fid | params | estrato | busca |
|---|---|---|---|---|---|---|---|
| **ConvNeXtV2-pico** (congelado) | 0.792 | — | — | — | — | FM | baseline |
| **I-JEPA** (congelado) | 0.785 | — | — | — | — | FM | baseline |
| **AIMv2** (congelado) | 0.737 | — | — | — | — | FM | baseline |
| **DINOv2** (congelado) | 0.685 | — | — | — | — | FM | baseline |
| **ResNet-50** (congelado) | 0.671 | — | — | — | — | FM | baseline |
| c00001 | 0.815 | 0.692 | 0.646 | 100 | 22k | tiny | hybrid C100_kw1 (modehb_hybrid) |
| c00001 | 0.813 | 0.692 | 0.646 | 200 | 22k | tiny | hybrid C100_kw1 (modehb_hybrid) |
| c00003 | 0.807 | 0.763 | 0.751 | 200 | 991k | medium | hybrid C100_kw1 (modehb_hybrid) |
| c00003 | 0.806 | 0.758 | 0.766 | 100 | 991k | medium | hybrid C100_kw1 (modehb_hybrid) |
| c00004 | 0.768 | 0.790 | 0.790 | 100 | 2298k | huge | hybrid C100_kw1 (modehb_hybrid) |
| c00002 | 0.754 | 0.736 | 0.763 | 200 | 474k | small | hybrid C100_kw1 (modehb_hybrid) |
| c00002 | 0.744 | 0.738 | 0.750 | 45 | 474k | small | hybrid C100_kw1 (modehb_hybrid) |
| c00004 | 0.737 | 0.787 | 0.783 | 200 | 2298k | huge | hybrid C100_kw1 (modehb_hybrid) |
| c00004 | 0.735 | 0.761 | 0.778 | 200 | 2223k | huge | lemonade C100_kw1 (modehb_lemonade) |
| c00002 | 0.732 | 0.753 | 0.763 | 100 | 474k | small | hybrid C100_kw1 (modehb_hybrid) |
| c00001 | 0.715 | 0.692 | 0.732 | 200 | 201k | tiny | lemonade C100_kw1 (modehb_lemonade) |
| c00004 | 0.704 | 0.755 | 0.787 | 100 | 2223k | huge | lemonade C100_kw1 (modehb_lemonade) |

## eggs

| modelo | κ teste SVM original | κ teste cov_head | κ val (fitness) | fid | params | estrato | busca |
|---|---|---|---|---|---|---|---|
| **DINOv2** (congelado) | 0.988 | — | — | — | — | FM | baseline |
| **AIMv2** (congelado) | 0.985 | — | — | — | — | FM | baseline |
| **I-JEPA** (congelado) | 0.976 | — | — | — | — | FM | baseline |
| **ResNet-50** (congelado) | 0.948 | — | — | — | — | FM | baseline |
| **ConvNeXtV2-pico** (congelado) | 0.944 | — | — | — | — | FM | baseline |
| c00002 | 0.852 | 0.881 | 0.875 | 200 | 562k | small | lemonade C100_kw1 (modehb_lemonade) |
| c00002 | 0.834 | 0.867 | 0.882 | 200 | 562k | small | hybrid C100_kw1 (modehb_hybrid) |
| c00004 | 0.827 | 0.962 | 0.932 | 200 | 1697k | huge | hybrid C100_kw1 (modehb_hybrid) |
| c00004 | 0.794 | 0.959 | 0.931 | 100 | 1697k | huge | hybrid C100_kw1 (modehb_hybrid) |
| c00003 | 0.747 | 0.959 | 0.910 | 100 | 1342k | medium | hybrid C100_kw1 (modehb_hybrid) |
| c00003 | 0.744 | 0.953 | 0.910 | 200 | 1342k | medium | hybrid C100_kw1 (modehb_hybrid) |
| c00079 | 0.705 | 0.891 | 0.869 | 200 | 838k | medium | lemonade C100_kw1 (modehb_lemonade) |
| c00003 | 0.580 | 0.958 | 0.906 | 200 | 1251k | medium | lemonade C100_kw1 (modehb_lemonade) |
| c00004 | 0.298 | 0.941 | 0.904 | 200 | 2141k | huge | lemonade C100_kw1 (modehb_lemonade) |

## eggs_raw

| modelo | κ teste SVM original | κ teste cov_head | κ val (fitness) | fid | params | estrato | busca |
|---|---|---|---|---|---|---|---|
| c00004 | 0.682 | 0.922 | 0.896 | 200 | 1697k | huge | hybrid C100_kw1 (modehb_hybrid) |
| c00002 | 0.647 | 0.790 | 0.753 | 100 | 562k | small | hybrid C100_kw1 (modehb_hybrid) |
| c00002 | 0.612 | 0.798 | 0.753 | 200 | 562k | small | hybrid C100_kw1 (modehb_hybrid) |
| c00001 | 0.521 | 0.565 | 0.527 | 200 | 13k | tiny | hybrid C100_kw1 (modehb_hybrid) |
| c00003 | 0.220 | 0.922 | 0.878 | 200 | 1342k | medium | hybrid C100_kw1 (modehb_hybrid) |

## flowers

| modelo | κ teste SVM original | κ teste cov_head | κ val (fitness) | fid | params | estrato | busca |
|---|---|---|---|---|---|---|---|
| **AIMv2** (congelado) | 0.979 | — | — | — | — | FM | baseline |
| **DINOv2** (congelado) | 0.976 | — | — | — | — | FM | baseline |
| **I-JEPA** (congelado) | 0.930 | — | — | — | — | FM | baseline |
| **ConvNeXtV2-pico** (congelado) | 0.878 | — | — | — | — | FM | baseline |
| **ResNet-50** (congelado) | 0.872 | — | — | — | — | FM | baseline |
| c00043 | 0.754 | 0.805 | 0.739 | 100 | 284k | tiny | lemonade C100_kw1 (modehb_lemonade) |
| c00043 | 0.745 | 0.836 | 0.769 | 200 | 284k | tiny | lemonade C100_kw1 (modehb_lemonade) |
| c00043 | 0.678 | 0.793 | 0.750 | 45 | 284k | tiny | lemonade C100_kw1 (modehb_lemonade) |
| c00043 | 0.668 | 0.802 | 0.793 | 15 | 1119k | medium | hybrid C100_kw1 (modehb_hybrid) |
| c00035 | 0.665 | 0.814 | 0.754 | 5 | 1239k | medium | hybrid C100_kw1 (modehb_hybrid) |
| c00035 | 0.650 | 0.796 | 0.754 | 15 | 1239k | medium | hybrid C100_kw1 (modehb_hybrid) |
| c00001 | 0.638 | 0.619 | 0.653 | 100 | 21k | tiny | hybrid C100_kw1 (modehb_hybrid) |
| c00043 | 0.620 | 0.781 | 0.796 | 5 | 1119k | medium | hybrid C100_kw1 (modehb_hybrid) |
| c00077 | 0.616 | 0.796 | 0.769 | 45 | 2495k | huge | lemonade C100_kw1 (modehb_lemonade) |
| c00004 | 0.604 | 0.774 | 0.772 | 200 | 1887k | huge | hybrid C100_kw1 (modehb_hybrid) |
| c00003 | 0.591 | 0.781 | 0.744 | 200 | 1312k | medium | lemonade C100_kw1 (modehb_lemonade) |
| c00075 | 0.586 | 0.799 | 0.756 | 45 | 473k | small | lemonade C100_kw1 (modehb_lemonade) |

## ham10000

| modelo | κ teste SVM original | κ teste cov_head | κ val (fitness) | fid | params | estrato | busca |
|---|---|---|---|---|---|---|---|
| **AIMv2** (congelado) | 0.724 | — | — | — | — | FM | baseline |
| **DINOv2** (congelado) | 0.701 | — | — | — | — | FM | baseline |
| **I-JEPA** (congelado) | 0.595 | — | — | — | — | FM | baseline |
| **ResNet-50** (congelado) | 0.492 | — | — | — | — | FM | baseline |
| **ConvNeXtV2-pico** (congelado) | 0.485 | — | — | — | — | FM | baseline |
| c00061 | 0.588 | 0.636 | 0.601 | 45 | 1046k | medium | lemonade C100_kw1 (modehb_lemonade) |
| c00045 | 0.586 | nan | 0.582 | 45 | 847k | medium | lemonade C100_kw1 (modehb_lemonade) |
| c00089 | 0.574 | 0.624 | 0.604 | 100 | 1085k | medium | lemonade C100_kw1 (modehb_lemonade) |
| c00089 | 0.567 | 0.618 | 0.620 | 45 | 1085k | medium | lemonade C100_kw1 (modehb_lemonade) |
| c00085 | 0.556 | nan | 0.575 | 45 | 928k | medium | lemonade C100_kw1 (modehb_lemonade) |
| c00112 | 0.554 | nan | 0.581 | 45 | 915k | medium | lemonade C100_kw1 (modehb_lemonade) |
| c00070 | 0.542 | 0.613 | 0.583 | 45 | 1532k | huge | lemonade C100_kw1 (modehb_lemonade) |
| c00070 | 0.537 | 0.607 | 0.585 | 100 | 1532k | huge | lemonade C100_kw1 (modehb_lemonade) |
| c00070 | 0.537 | 0.614 | 0.564 | 200 | 1532k | huge | lemonade C100_kw1 (modehb_lemonade) |
| c00001 | 0.534 | 0.480 | 0.463 | 200 | 20k | tiny | hybrid C100_kw1 (modehb_hybrid) |
| c00113 | 0.526 | 0.616 | 0.572 | 45 | 1532k | huge | lemonade C100_kw1 (modehb_lemonade) |
| c00113 | 0.525 | 0.606 | 0.584 | 100 | 1532k | huge | lemonade C100_kw1 (modehb_lemonade) |

## larvae

| modelo | κ teste SVM original | κ teste cov_head | κ val (fitness) | fid | params | estrato | busca |
|---|---|---|---|---|---|---|---|
| **AIMv2** (congelado) | 1.000 | — | — | — | — | FM | baseline |
| **DINOv2** (congelado) | 1.000 | — | — | — | — | FM | baseline |
| **ConvNeXtV2-pico** (congelado) | 0.933 | — | — | — | — | FM | baseline |
| **I-JEPA** (congelado) | 0.919 | — | — | — | — | FM | baseline |
| **ResNet-50** (congelado) | 0.906 | — | — | — | — | FM | baseline |
| c00003 | 0.858 | 0.967 | 0.935 | 200 | 1341k | medium | lemonade C100_kw1 (modehb_lemonade) |
| c00002 | 0.858 | 0.851 | 0.917 | 200 | 594k | small | lemonade C100_kw1 (modehb_lemonade) |
| c00042 | 0.814 | 0.897 | 0.866 | 100 | 453k | small | hybrid C100_kw1 (modehb_hybrid) |
| c00002 | 0.808 | 0.900 | 0.902 | 100 | 594k | small | lemonade C100_kw1 (modehb_lemonade) |
| c00002 | 0.780 | 0.933 | 0.917 | 200 | 594k | small | hybrid C100_kw1 (modehb_hybrid) |
| c00002 | 0.780 | 0.933 | 0.917 | 100 | 594k | small | hybrid C100_kw1 (modehb_hybrid) |
| c00051 | 0.760 | 0.851 | 0.889 | 200 | 1431k | medium | hybrid C100_kw1 (modehb_hybrid) |
| c00075 | 0.758 | 0.917 | 0.902 | 100 | 425k | small | lemonade C100_kw1 (modehb_lemonade) |
| c00003 | 0.752 | 0.882 | 0.950 | 200 | 1355k | medium | hybrid C100_kw1 (modehb_hybrid) |
| c00003 | 0.752 | 0.882 | 0.950 | 100 | 1355k | medium | hybrid C100_kw1 (modehb_hybrid) |
| c00004 | 0.750 | 0.900 | 0.933 | 200 | 1974k | huge | hybrid C100_kw1 (modehb_hybrid) |
| c00007 | 0.725 | 0.882 | 0.902 | 5 | 1974k | huge | lemonade C100_kw1 (modehb_lemonade) |

## larvae_raw

| modelo | κ teste SVM original | κ teste cov_head | κ val (fitness) | fid | params | estrato | busca |
|---|---|---|---|---|---|---|---|
| c00065 | 0.663 | 0.810 | 0.879 | 200 | 225k | tiny | hybrid C100_kw1 (modehb_hybrid) |
| c00003 | 0.454 | 0.740 | 0.810 | 200 | 1355k | medium | hybrid C100_kw1 (modehb_hybrid) |
| c00004 | 0.401 | 0.829 | 0.848 | 100 | 1974k | huge | hybrid C100_kw1 (modehb_hybrid) |
| c00002 | 0.357 | 0.509 | 0.627 | 200 | 594k | small | hybrid C100_kw1 (modehb_hybrid) |
| c00004 | 0.190 | 0.810 | 0.879 | 200 | 1974k | huge | hybrid C100_kw1 (modehb_hybrid) |
| c00004 | 0.188 | 0.740 | 0.915 | 45 | 1974k | huge | hybrid C100_kw1 (modehb_hybrid) |

## protozoan

| modelo | κ teste SVM original | κ teste cov_head | κ val (fitness) | fid | params | estrato | busca |
|---|---|---|---|---|---|---|---|
| **AIMv2** (congelado) | 0.968 | — | — | — | — | FM | baseline |
| **DINOv2** (congelado) | 0.956 | — | — | — | — | FM | baseline |
| **I-JEPA** (congelado) | 0.938 | — | — | — | — | FM | baseline |
| **ConvNeXtV2-pico** (congelado) | 0.853 | — | — | — | — | FM | baseline |
| **ResNet-50** (congelado) | 0.822 | — | — | — | — | FM | baseline |
| c00002 | 0.843 | 0.921 | 0.839 | 200 | 678k | small | hybrid C100_kw1 (modehb_hybrid) |
| c00001 | 0.798 | 0.792 | 0.674 | 45 | 20k | tiny | hybrid C100_kw1 (modehb_hybrid) |
| c00002 | 0.796 | 0.908 | 0.838 | 5 | 678k | small | hybrid C100_kw1 (modehb_hybrid) |
| c00001 | 0.790 | 0.803 | 0.690 | 45 | 20k | tiny | hybrid C0.1_kw0.75 (modehb_hybrid_C01_2049) |
| c00001 | 0.788 | 0.786 | 0.670 | 100 | 20k | tiny | hybrid C100_kw1 (modehb_hybrid) |
| c00001 | 0.784 | 0.791 | 0.680 | 15 | 20k | tiny | hybrid C100_kw1 (modehb_hybrid) |
| c00001 | 0.778 | 0.783 | 0.666 | 200 | 20k | tiny | hybrid C100_kw1 (modehb_hybrid) |
| c00001 | 0.772 | 0.767 | 0.681 | 100 | 20k | tiny | hybrid C0.1_kw0.75 (modehb_hybrid_C01_2049) |
| c00004 | 0.753 | 0.888 | 0.808 | 200 | 1729k | huge | hybrid C100_kw1 (modehb_hybrid) |
| c00003 | 0.706 | 0.941 | 0.889 | 200 | 875k | medium | hybrid C100_kw1 (modehb_hybrid) |

## protozoan_raw

| modelo | κ teste SVM original | κ teste cov_head | κ val (fitness) | fid | params | estrato | busca |
|---|---|---|---|---|---|---|---|
| c00004 | 0.663 | 0.890 | 0.785 | 200 | 1812k | huge | hybrid C100_kw1 (modehb_hybrid) |
| c00002 | 0.554 | 0.708 | 0.630 | 200 | 620k | small | hybrid C100_kw1 (modehb_hybrid) |
| c00002 | 0.554 | nan | 0.630 | 100 | 620k | small | hybrid C100_kw1 (modehb_hybrid) |
| c00003 | 0.452 | 0.874 | 0.796 | 200 | 1307k | medium | hybrid C100_kw1 (modehb_hybrid) |

## remote_sensing

| modelo | κ teste SVM original | κ teste cov_head | κ val (fitness) | fid | params | estrato | busca |
|---|---|---|---|---|---|---|---|
| **AIMv2** (congelado) | 0.746 | — | — | — | — | FM | baseline |
| **DINOv2** (congelado) | 0.719 | — | — | — | — | FM | baseline |
| **ResNet-50** (congelado) | 0.704 | — | — | — | — | FM | baseline |
| **I-JEPA** (congelado) | 0.698 | — | — | — | — | FM | baseline |
| **ConvNeXtV2-pico** (congelado) | 0.655 | — | — | — | — | FM | baseline |
| c00041 | 0.604 | 0.692 | 0.590 | 45 | 2275k | large | lemonade_seeded C100_kw1 (modehb_lemonade_seeded) |
| c00098 | 0.598 | 0.637 | 0.543 | 15 | 531k | small | lemonade C0.1_kw0.75 (modehb_lemonade_C01_final) |
| c00010 | 0.585 | 0.627 | 0.571 | 45 | 246k | tiny | hybrid C0.1_kw0.75 (modehb_hybrid_C01_final) |
| c00044 | 0.583 | 0.611 | 0.536 | 45 | 47k | small | lemonade_seeded C100_kw1 (modehb_lemonade_seeded) |
| c00086 | 0.577 | 0.639 | 0.538 | 45 | 54k | tiny | lemonade C0.1_kw0.75 (modehb_lemonade_C01_final) |
| c00015 | 0.576 | 0.685 | 0.576 | 45 | 3136k | xl | lemonade_seeded C100_kw1 (modehb_lemonade_seeded) |
| c00015 | 0.576 | 0.685 | 0.583 | 15 | 3136k | xl | lemonade_seeded C100_kw1 (modehb_lemonade_seeded) |
| c00002 | 0.571 | 0.666 | 0.591 | 200 | 1384k | medium | lemonade_seeded C100_kw1 (modehb_lemonade_seeded) |
| c00093 | 0.565 | 0.575 | 0.536 | 15 | 3068k | huge | lemonade C0.1_kw0.75 (modehb_lemonade_C01_final) |
| c00002 | 0.563 | 0.678 | 0.577 | 100 | 1384k | medium | lemonade_seeded C100_kw1 (modehb_lemonade_seeded) |
| c00004 | 0.558 | 0.658 | 0.556 | 200 | 4728k | xl | lemonade_seeded C100_kw1 (modehb_lemonade_seeded) |
| c00004 | 0.558 | 0.658 | 0.556 | 100 | 4728k | xl | lemonade_seeded C100_kw1 (modehb_lemonade_seeded) |
