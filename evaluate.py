#!/usr/bin/env python3
"""Evaluate a trained model on fixed test mixtures.

Metrics (per 2-speaker utterance, PIT-assigned):
  - SI-SNRi, SDRi        (primary; dB improvement over the mixture)
  - PESQ, STOI           (perceptual, computed at 16 kHz on CPU)
  - inference latency (ms) on the requested device(s), peak GPU memory (MiB)

Usage:
    python evaluate.py --config configs/train_bimamba_f0.yaml \
        --checkpoint runs/bimamba-f0/checkpoints/best.pt \
        --mixtures mixtures/test --perceptual --latency-repeats 20

Without --checkpoint the random-init model is evaluated (sanity baseline,
SI-SNRi should be ~0 dB).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from conformer_bimamba.data import FixedMixtureDataset
from conformer_bimamba.models import HybridSepformer
from conformer_bimamba.utils.config import load_config
from conformer_bimamba.utils.metrics import (
    compute_perceptual,
    measure_latency,
    sdri,
    si_snri,
)
from conformer_bimamba.utils.seed import set_seed

from train import build_model  # reuse config->model construction


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", action="append", default=[], help="dotted.key=value (repeatable)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--mixtures", default=None, help="override data.test_mixtures")
    ap.add_argument("--perceptual", action="store_true", help="compute PESQ/STOI")
    ap.add_argument("--latency-repeats", type=int, default=20)
    ap.add_argument("--cpu-latency", action="store_true", help="also time on CPU (reference scan)")
    ap.add_argument("--out-json", default=None, help="save a JSON summary")
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    set_seed(cfg.get("seed", 42))
    m = cfg["model"]
    mixtures = args.mixtures or cfg["data"]["test_mixtures"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[eval] loaded {args.checkpoint} (best SI-SNRi {ckpt.get('best_si_snri', 'n/a')})")
    else:
        print("[eval] WARNING: no checkpoint — evaluating random-init model")
    model.eval()

    loader = DataLoader(FixedMixtureDataset(mixtures), batch_size=1, shuffle=False, num_workers=2)

    agg = {"si_snri": [], "sdri": [], "pesq": [], "stoi": []}
    for batch in loader:
        mixture = batch["mixture"].to(device)
        sources = batch["sources"].to(device)
        f0 = batch["f0"].squeeze(-1).to(device)
        est = model(mixture, f0)
        T = min(est.shape[-1], sources.shape[-1])
        est, ref = est[..., :T].squeeze(2), sources[..., :T]
        agg["si_snri"].append(si_snri(est, ref, mixture).item())
        agg["sdri"].append(sdri(est, ref, mixture).item())
        if args.perceptual:
            p = compute_perceptual(est.unsqueeze(0), ref.unsqueeze(0), cfg["data"]["sample_rate"])
            agg["pesq"].append(p["pesq"])
            agg["stoi"].append(p["stoi"])

    import statistics

    summary = {
        "n_utterances": len(agg["si_snri"]),
        "si_snri_mean_db": statistics.mean(agg["si_snri"]),
        "si_snri_std_db": statistics.stdev(agg["si_snri"]) if len(agg["si_snri"]) > 1 else 0.0,
        "sdri_mean_db": statistics.mean(agg["sdri"]),
        "params_m": model.num_parameters() / 1e6,
    }
    if args.perceptual:
        summary["pesq_mean"] = statistics.mean([v for v in agg["pesq"] if v == v])  # skip NaN
        summary["stoi_mean"] = statistics.mean([v for v in agg["stoi"] if v == v])

    # --- latency -----------------------------------------------------------
    sample = next(iter(loader))
    lat = measure_latency(
        model,
        {
            "mixture": sample["mixture"].to(device),
            "f0": sample["f0"].squeeze(-1).to(device),
        },
        device,
        repeats=args.latency_repeats,
    )
    summary.update({f"{device.type}_{k}": v for k, v in lat.items()})

    if args.cpu_latency:
        cpu_model = build_model(cfg).to("cpu")
        if args.checkpoint:
            cpu_model.load_state_dict(
                torch.load(args.checkpoint, map_location="cpu")["model_state"]
            )
        cpu_model.eval()
        cpu_lat = measure_latency(
            cpu_model,
            {
                "mixture": sample["mixture"].cpu(),
                "f0": sample["f0"].squeeze(-1).cpu(),
            },
            torch.device("cpu"),
            repeats=min(args.latency_repeats, 5),
        )
        summary.update({f"cpu_{k}": v for k, v in cpu_lat.items()})

    print(json.dumps(summary, indent=2))
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(summary, indent=2))
        print(f"[eval] summary -> {args.out_json}")


if __name__ == "__main__":
    main()
