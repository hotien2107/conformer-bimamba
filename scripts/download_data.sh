#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-shot data preparation for the Vietnamese speech separation experiments.
# Steps: (1) download Vivos (+ optionally Common Voice vi) and build
# manifests, (2) extract F0 offline, (3) build fixed val/test mixtures.
#
# Requires: python3 + venv (see requirements.txt). Set HF_TOKEN if you want
# Common Voice (gated). Run from the repository root.
#
#   bash scripts/download_data.sh [--with_commonvoice]
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

WITH_CV="${1:-}"
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

echo "==> 1/4 Download corpora + build manifests"
"$PY" -m conformer_bimamba.data.download --out_dir data --f0_root data/f0 $WITH_CV

echo "==> 2/4 Extract F0 (PyWorld, offline, parallel)"
"$PY" -m conformer_bimamba.preprocess.extract_f0 \
    --manifest data/manifests/vivos_train.jsonl --out_dir data/f0 \
    --out_manifest data/manifests/vivos_train_f0.jsonl --workers 8

# If Common Voice is enabled, add its train split to the training manifest.
if [ -f data/manifests/commonvoice_vi_train.jsonl ]; then
  echo "==> 2b  extract F0 for Common Voice"
  "$PY" -m conformer_bimamba.preprocess.extract_f0 \
      --manifest data/manifests/commonvoice_vi_train.jsonl --out_dir data/f0 \
      --out_manifest data/manifests/commonvoice_vi_train_f0.jsonl --workers 8
  # merge the two train manifests (train.py accepts one manifest; concatenate)
  cat data/manifests/vivos_train_f0.jsonl data/manifests/commonvoice_vi_train_f0.jsonl \
      > data/manifests/train_f0.jsonl
else
  cp data/manifests/vivos_train_f0.jsonl data/manifests/train_f0.jsonl
fi

echo "==> 3/4 Build fixed validation mixtures (dynamic mixing for train happens online)"
"$PY" -m conformer_bimamba.preprocess.make_mixtures \
    --manifest data/manifests/vivos_test.jsonl --out_dir mixtures/val \
    --n_pairs 300 --sr 32000 --dur_sec 5.0 --snr_db 0 --seed 0 --workers 8

echo "==> 4/4 Build fixed test mixtures (bigger, held-out)"
"$PY" -m conformer_bimamba.preprocess.make_mixtures \
    --manifest data/manifests/vivos_test.jsonl --out_dir mixtures/test \
    --n_pairs 500 --sr 32000 --dur_sec 5.0 --snr_db 0 --seed 1 --workers 8

echo "==> done."
echo "Next: point configs/*.yaml data.train_manifest at data/manifests/train_f0.jsonl"
echo "      (or pass --override data.train_manifest=...), then run train.py"
