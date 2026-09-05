"""Generate fixed, reproducible 2-speaker mixtures for val/test.

These are *offline* mixtures (as opposed to the dynamic mixing used during
training) so that every run of ``evaluate.py`` scores the exact same audio.
F0 is extracted from the actual mixture with PyWorld — the honest input the
model will see at inference time.

Usage:
    python -m conformer_bimamba.preprocess.make_mixtures \
        --manifest manifests/test.jsonl --out_dir mixtures/test \
        --n_pairs 500 --sr 32000 --dur_sec 5.0 --snr_db 0 --seed 0 --workers 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from ..data.vivos import load_manifest
from ..utils.audio import load_wav, make_mixture
from .extract_f0 import extract_f0_from_wav


def _make_pair(args: tuple) -> tuple[str, dict]:
    pair_id, e1, e2, out_root, sr, dur_sec, snr_db, seed = args
    rng = random.Random(seed)
    target_len = int(dur_sec * sr)
    s1 = load_wav(e1["path"], sr)
    s2 = load_wav(e2["path"], sr)
    mixture, s1, s2 = make_mixture(s1, s2, snr_db, target_len)

    out = Path(out_root) / f"pair_{pair_id:05d}"
    out.mkdir(parents=True, exist_ok=True)
    for name, wav in (("mixture", mixture), ("s1", s1), ("s2", s2)):
        sf.write(str(out / f"{name}.wav"), wav.squeeze(0).numpy(), sr)

    # F0 of the *actual mixture* (100 fps, Hz, 0 = unvoiced)
    f0 = extract_f0_from_wav(mixture.squeeze(0).numpy(), sr)
    np.save(str(out / "f0.npy"), f0)

    meta = {
        "pair_id": pair_id,
        "speaker_1": e1["speaker"],
        "speaker_2": e2["speaker"],
        "path_1": e1["path"],
        "path_2": e2["path"],
        "snr_db": snr_db,
        "sample_rate": sr,
        "duration_sec": dur_sec,
    }
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return str(out), meta


def make_fixed_mixtures(
    entries: list[dict],
    out_root: str | Path,
    n_pairs: int = 500,
    sr: int = 32000,
    dur_sec: float = 5.0,
    snr_db: float = 0.0,
    seed: int = 0,
    workers: int = 8,
) -> list[dict]:
    """Create ``n_pairs`` fixed mixtures from disjoint-speaker pairs."""
    rng = random.Random(seed)
    by_speaker: dict[str, list[dict]] = {}
    for e in entries:
        by_speaker.setdefault(e["speaker"], []).append(e)
    speakers = list(by_speaker)

    pairs = []
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 20:
        attempts += 1
        s1 = rng.choice(speakers)
        s2 = rng.choice([s for s in speakers if s != s1])
        e1 = rng.choice(by_speaker[s1])
        e2 = rng.choice(by_speaker[s2])
        pairs.append((len(pairs), e1, e2, str(out_root), sr, dur_sec, snr_db, rng.randrange(2**31)))

    jobs = list(pairs)  # each: (pair_id, e1, e2, out_root, sr, dur_sec, snr_db, seed)
    metas = []
    with mp.Pool(workers) as pool:
        for i, (path, meta) in enumerate(pool.imap_unordered(_make_pair, jobs, chunksize=4)):
            metas.append(meta)
            if (i + 1) % 50 == 0:
                print(f"  mixtures: {i + 1}/{n_pairs}", flush=True)

    manifest_path = Path(out_root) / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)
    print(f"[make_mixtures] {len(metas)} mixtures -> {out_root}")
    return metas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_pairs", type=int, default=500)
    ap.add_argument("--sr", type=int, default=32000)
    ap.add_argument("--dur_sec", type=float, default=5.0)
    ap.add_argument("--snr_db", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    entries = load_manifest(args.manifest)
    make_fixed_mixtures(
        entries, args.out_dir, args.n_pairs, args.sr, args.dur_sec, args.snr_db, args.seed, args.workers
    )


if __name__ == "__main__":
    sys.exit(main())
