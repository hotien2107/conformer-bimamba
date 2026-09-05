#!/bin/sh
# Train launcher for train/speech_separation (ClearerVoice-Studio style).
#
# Reads the same environment variables as the Google-Colab notebooks
# (CONFIG_PTH / CHECKPOINT_DIR / ...) so the pipeline is driven identically to
# the MossFormer2 sample; falls back to the hard-coded defaults below when the
# envs are not set (e.g. when run directly in the terminal).
#
#   USE (Colab): set the envs then `bash train.sh`, or edit the defaults below.
#
# Usage:              ./train.sh                       # = ./train.sh 16K single GPU
#   N_GPU=2 ./train.sh                                # 2 GPUs (distributed)

# ---- environment (overridable) ----
GPU_ID="${GPU_ID:-0}"                       # visible GPU ids
N_GPU="${N_GPU:-1}"                         # number of GPUs / world size
NETWORK="${NETWORK:-HybridBiMamba_SS_16K}"  # network name
CONFIG_PTH="${CONFIG_PTH:-config/train/${NETWORK}.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/${NETWORK}}"
TRAIN_FROM_LAST_CHECKPOINT="${TRAIN_FROM_LAST_CHECKPOINT:-0}"
INIT_CHECKPOINT_PATH="${INIT_CHECKPOINT_PATH:-None}"
PRINT_FREQ="${PRINT_FREQ:-10}"
CHECKPOINT_SAVE_FREQ="${CHECKPOINT_SAVE_FREQ:-5000}"

if [ ! -d "${CHECKPOINT_DIR}" ]; then
  mkdir -p "${CHECKPOINT_DIR}"
fi
cp "${CONFIG_PTH}" "${CHECKPOINT_DIR}/config.yaml" 2>/dev/null || true

export PYTHONWARNINGS="ignore"

if [ "${N_GPU}" -gt 1 ]; then
  # distributed (same launcher as the original ClearerVoice-Studio)
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  python -W ignore \
    -m torch.distributed.launch \
    --nproc_per_node="${N_GPU}" \
    --master_port="$(date '+88%S')" \
    train.py \
    --config "${CONFIG_PTH}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --train_from_last_checkpoint "${TRAIN_FROM_LAST_CHECKPOINT}" \
    --init_checkpoint_path "${INIT_CHECKPOINT_PATH}" \
    --print_freq "${PRINT_FREQ}" \
    --checkpoint_save_freq "${CHECKPOINT_SAVE_FREQ}"
else
  # single GPU / Colab
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  python -W ignore train.py \
    --config "${CONFIG_PTH}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --train_from_last_checkpoint "${TRAIN_FROM_LAST_CHECKPOINT}" \
    --init_checkpoint_path "${INIT_CHECKPOINT_PATH}" \
    --print_freq "${PRINT_FREQ}" \
    --checkpoint_save_freq "${CHECKPOINT_SAVE_FREQ}"
fi
