"""Offline F0 (pitch) extraction with PyWorld.

Precomputes per-utterance F0 contours so training never pays the pyworld
cost online. Output: one ``.npy`` per utterance (float32, Hz, 0 = unvoiced,
10 ms frames) + a manifest copy whose entries carry the ``f0`` path.

Usage:
    python -m conformer_bimamba.preprocess.extract_f0 \
        --manifest manifests/train.jsonl --out_dir f0_cache --workers 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

from ..data.vivos import load_manifest

F0_FLOOR = 60.0   # Hz — below typical Vietnamese male pitch
F0_CEIL = 400.0   # Hz — above typical Vietnamese female pitch (tones ride on this)
FRAME_PERIOD = 10.0  # ms -> 100 fps


def extract_f0_from_wav(
    wav: np.ndarray,
    sr: int,
    f0_floor: float = F0_FLOOR,
    f0_ceil: float = F0_CEIL,
    frame_period: float = FRAME_PERIOD,
) -> np.ndarray:
    """PyWorld DIO + Stonemask (smoothing). Returns (n_frames,) float32 Hz."""
    import pyworld

    x = wav.astype(np.float64).copy()
    if x.size == 0:
        return np.zeros(0, dtype=np.float32)
    f0, t = pyworld.dio(x, sr, f0_floor=f0_floor, f0_ceil=f0_ceil, frame_period=frame_period)
    f0 = pyworld.stonemask(x, f0, t, sr)
    return f0.astype(np.float32)


def _worker(args: tuple) -> dict:
    entry, f0_root, f0_floor, f0_ceil = args
    import soundfile as sf

    wav, sr = sf.read(entry["path"], dtype="float32", always_2d=False)
    f0 = extract_f0_from_wav(wav, sr, f0_floor, f0_ceil)
    out = Path(f0_root) / f"{Path(entry['path']).name}.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out), f0)
    e = dict(entry)
    e["f0"] = str(out)
    return e


def extract_manifest(entries: list[dict], f0_root: str | Path, workers: int = 8) -> list[dict]:
    f0_root = Path(f0_root)
    f0_root.mkdir(parents=True, exist_ok=True)
    jobs = [(e, str(f0_root), F0_FLOOR, F0_CEIL) for e in entries]
    with mp.Pool(workers) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(_worker, jobs, chunksize=16)):
            results.append(r)
            if (i + 1) % 500 == 0:
                print(f"  f0: {i + 1}/{len(jobs)}", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="input utterance manifest (jsonl)")
    ap.add_argument("--out_dir", required=True, help="f0 cache directory")
    ap.add_argument("--out_manifest", default=None, help="manifest copy with f0 paths (default: <out_dir>/manifest.jsonl)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    entries = load_manifest(args.manifest)
    print(f"[extract_f0] processing {len(entries)} utterances with {args.workers} workers")
    results = extract_manifest(entries, args.out_dir, args.workers)

    out_manifest = args.out_manifest or str(Path(args.out_dir) / "manifest.jsonl")
    from ..data.vivos import save_manifest

    save_manifest(results, out_manifest)
    print(f"[extract_f0] done -> {out_manifest}")


if __name__ == "__main__":
    sys.exit(main())
