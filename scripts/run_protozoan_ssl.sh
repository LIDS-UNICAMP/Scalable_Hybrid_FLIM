#!/usr/bin/env bash
# run_protozoan_ssl.sh — Run all 72 protozoan-cysts SSL pre-training experiments.
# Architecture: ch24_30_48 (30 conv2 channels — matching FLIM kernel selection).
# All 4 initializations: xavier, random, he, flim.
# All 3 splits, all 6 percentages.
#
# Usage (sequential, one GPU):
#   bash scripts/run_protozoan_ssl.sh
#
# Usage (parallel, N GPUs via GNU parallel):
#   cat scripts/run_protozoan_ssl.sh | grep "^python" | \
#     parallel -j N --delay 10 OMP_NUM_THREADS=4 {}

set -euo pipefail

SPLITS=(1 2 3)
PCTS=(1 5 25 50 75 100)
INITS=(xavier random he flim)

for split in "${SPLITS[@]}"; do
  for pct in "${PCTS[@]}"; do
    for init in "${INITS[@]}"; do
      if [ "$init" = "flim" ]; then
        model_cfg="configs/model/lejepa_line_flim_protozoan_train${split}.yaml"
      else
        model_cfg="configs/model/lejepa_line_${init}_protozoan_train${split}.yaml"
      fi

      echo ">>> split=${split} pct=${pct} init=${init}"
      OMP_NUM_THREADS=4 python src/main.py fit \
        --config configs/default.yaml \
        --config "configs/data/percentage/protozoan-cysts_split_${split}/${pct}/lejepa_line.yaml" \
        --config "$model_cfg" \
        --trainer.accelerator=gpu \
        --trainer.devices=1
    done
  done
done

echo "All 72 protozoan experiments done."
