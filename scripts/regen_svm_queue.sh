#!/usr/bin/env bash
# regen_svm_queue.sh — Regenera os SVMs em FILA (um por vez), presos a UMA GPU,
# com prioridade baixa de CPU para NÃO atrapalhar o treino que já está rodando.
#
# Uso (roda JÁ, em paralelo ao treino, na GPU 0):
#   tmux new-session -d -s svm_regen 'bash scripts/regen_svm_queue.sh'
#
#   # escolher outra GPU:            bash scripts/regen_svm_queue.sh 1
#   # esperar um treino terminar:    bash scripts/regen_svm_queue.sh 0 distill_flim_600_800
#
# Acompanhar:  tail -f logs/svm_regen_*.log
set -uo pipefail   # NÃO usar -e: continuar mesmo se um SVM falhar

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${1:-0}"   # GPU dedicada aos SVMs (default 0)
WAIT_SESSION="${2:-}"                    # opcional: sessão tmux de treino a esperar

# Baixa prioridade de CPU + poucas threads → não rouba CPU do treino
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
RUN_PREFIX="nice -n 19"

LOG="logs/svm_regen_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

if [ -n "$WAIT_SESSION" ]; then
    echo "[$(date)] Aguardando treino '$WAIT_SESSION' terminar..."
    while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do sleep 300; done
    echo "[$(date)] Treino terminou."
fi
echo "[$(date)] Rodando SVMs em fila na GPU $CUDA_VISIBLE_DEVICES (nice 19, $OMP_NUM_THREADS threads)."

run() { echo; echo "[$(date)] >>> $*"; $RUN_PREFIX "$@"; echo "[$(date)] <<< rc=$? : $*"; }

# ── Leves (cabem com folga, rodam primeiro) ──────────────────────────────────
run python -m src.evaluate.unified_eval --model svm --dataset all
run python -m src.evaluate.svm_distillation_conv
# Backbone init-FLIM 400k (two-layer 1x1 256->1280), treinado com --no-imagenet-norm.
# Encoder 48d; --no-imagenet-norm casa o transform de teste com o treino.
run python -m src.evaluate.svm_distillation_conv --run-filter 2l_1x1_init_flim_256_1280_no_imagenet_norm --no-imagenet-norm --only-ok --output-csv svm_2l_1x1_init_flim_256_1280_nonorm_encoder48
run python -m src.evaluate.svm_distill_with_projection --output-csv svm_distill_proj1280_results
run python -m src.evaluate.svm_distill_with_projection --run-filter 3x3_BN2d_1280_one_layer --output-csv svm_proj1280_3x3_BN2d_results
run python -m src.evaluate.svm_distill_with_projection --run-filter 1x1_BN2d_1280_one_layer --output-csv svm_proj1280_1x1_BN2d_results
run python -m src.evaluate.svm_distill_with_projection --run-filter 2l_1x1_BN2d_256_1280

# ── Pesado (teacher I-JEPA ~2.4 GB) — por último, quando os outros já liberaram ─
run python -m src.evaluate.svm_ijepa

# ── Normalizar + plotar ──────────────────────────────────────────────────────
run python scripts/normalize_reports.py
run python scripts/plot_comparison_flim.py

echo "[$(date)] FIM da fila."
