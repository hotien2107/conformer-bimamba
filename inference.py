#!/usr/bin/env python3
"""Separate a wav file or a folder of mixtures with a trained checkpoint.

F0 is extracted on the fly with PyWorld when the model uses F0 conditioning
(the same pipeline as preprocess/extract_f0.py).

Usage:
    python inference.py --config configs/train_bimamba_f0.yaml \
        --checkpoint runs/bimamba-f0/checkpoints/best.pt \
        --input audio/mixture.wav --out_dir separated --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from conformer_bimamba.models import HybridSepformer
from conformer_bimamba.preprocess.extract_f0 import extract_f0_from_wav
from conformer_bimamba.utils.audio import load_wav, write_wav
from train import build_model


def separate_one(
    model: HybridSepformer,
    wav_path: str,
    out_dir: str | Path,
    sample_rate: int,
    device: torch.device,
    use_f0: bool,
) -> None:
    wav = load_wav(wav_path, sample_rate).to(device)  # (1, T)
    f0 = None
    if use_f0:
        try:
            from conformer_bimamba.preprocess.extract_f0 import extract_f0_from_wav

            import numpy as np

            f0_arr = extract_f0_from_wav(wav.squeeze(0).cpu().numpy(), sample_rate)
            f0 = torch.from_numpy(f0_arr).float().to(device)  # (L,)
        except ImportError:
            print(f"[infer] WARNING: pyworld not installed — running WITHOUT F0 conditioning "
                  f"({Path(wav_path).name})")

    model.eval()
    with torch.no_grad():
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            est = model(wav.unsqueeze(0), f0 if f0 is None else f0.unsqueeze(0))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(wav_path).stem
    est = est[0].squeeze(1).cpu()  # (2, T_dec)
    write_wav(str(out / f"{stem}_s1.wav"), est[0].unsqueeze(0), sample_rate)
    write_wav(str(out / f"{stem}_s2.wav"), est[1].unsqueeze(0), sample_rate)
    write_wav(str(out / f"{stem}_mixture.wav"), wav.cpu(), sample_rate)
    print(f"[infer] {Path(wav_path).name} -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", action="append", default=[], help="dotted.key=value (repeatable)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="wav file or directory of mixtures")
    ap.add_argument("--out_dir", default="separated")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    from conformer_bimamba.utils.config import load_config

    cfg = load_config(args.config, args.override)
    sample_rate = cfg["data"]["sample_rate"]

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[infer] loaded {args.checkpoint}")

    inp = Path(args.input)
    files = sorted(inp.rglob("*.wav")) if inp.is_dir() else [inp]
    for f in files:
        separate_one(model, str(f), args.out_dir, sample_rate, device, cfg["model"]["use_f0"])


if __name__ == "__main__":
    sys.exit(main())
