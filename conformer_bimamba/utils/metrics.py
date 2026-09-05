"""Separation metrics: SI-SNR(-i), SDR(-i), PESQ, STOI, latency."""
from __future__ import annotations

import time

import numpy as np
import torch

EPS = 1e-8


# ---------------------------------------------------------------------------
# SI-SNR (scale-invariant signal-to-noise ratio)
# ---------------------------------------------------------------------------
def si_snr(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """SI-SNR in dB. Both tensors (B, T). Returns (B,)."""
    assert estimate.shape == reference.shape
    target = (reference * estimate).sum(-1, keepdim=True) * reference / (
        reference.pow(2).sum(-1, keepdim=True) + EPS
    )
    noise = estimate - target
    return 10.0 * torch.log10(
        (target.pow(2).sum(-1) + EPS) / (noise.pow(2).sum(-1) + EPS)
    )


def si_snr_pit(estimates: torch.Tensor, references: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Permutation-invariant SI-SNR for 2 speakers.

    Args:
        estimates:  (B, 2, T) — the two separated streams.
        references: (B, 2, T) — the two reference sources.
    Returns:
        (best_si_snr, best_perm): best (B,) SI-SNR in dB and the (B,) best
        permutation index (0 -> identity, 1 -> swap).
    """
    b, s, t = estimates.shape
    assert references.shape == (b, s, t) and s == 2
    # (B, 2, 2, T) of si_snr(est_i, ref_j)
    e = estimates.unsqueeze(2).expand(b, s, s, t)     # (B, S, S, T) est_i vs all refs
    r = references.unsqueeze(1).expand(b, s, s, t)    # (B, S, S, T)
    scores = si_snr(e.reshape(-1, t), r.reshape(-1, t)).reshape(b, s, s)
    perm0 = scores.diagonal(dim1=1, dim2=2).sum(-1)   # identity
    perm1 = scores.flip(dims=(2,)).diagonal(dim1=1, dim2=2).sum(-1)  # swap
    best = torch.stack([perm0, perm1], dim=-1).max(dim=-1)
    return best.values, best.indices


def si_snr_loss(estimates: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
    """Negative mean PIT SI-SNR — the training loss (minimised)."""
    best, _ = si_snr_pit(estimates, references)
    return -best.mean()


# ---------------------------------------------------------------------------
# SDR (BSS-Eval style, naive least-squares projection)
# ---------------------------------------------------------------------------
def sdr(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Signal-to-distortion ratio in dB, (B,)."""
    target = (reference * estimate).sum(-1, keepdim=True) * reference / (
        reference.pow(2).sum(-1, keepdim=True) + EPS
    )
    distortion = estimate - target
    return 10.0 * torch.log10(
        (target.pow(2).sum(-1) + EPS) / (distortion.pow(2).sum(-1) + EPS)
    )


def sdr_pit(estimates: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
    b, s, t = estimates.shape
    e = estimates.unsqueeze(2).expand(b, s, s, t).reshape(-1, t)
    r = references.unsqueeze(1).expand(b, s, s, t).reshape(-1, t)
    scores = sdr(e, r).reshape(b, s, s)
    perm0 = scores.diagonal(dim1=1, dim2=2).sum(-1)
    perm1 = scores.flip(dims=(2,)).diagonal(dim1=1, dim2=2).sum(-1)
    return torch.stack([perm0, perm1], dim=-1).max(dim=-1).values


# ---------------------------------------------------------------------------
# Improvements over the mixture (SI-SNRi / SDRi), PIT-aware
# ---------------------------------------------------------------------------
def _improvement(
    score_fn, estimates: torch.Tensor, references: torch.Tensor, mixture: torch.Tensor
) -> torch.Tensor:
    """Per-utterance mean improvement (dB) after PIT assignment.

    estimates (B, 2, T), references (B, 2, T), mixture (B, 1, T).
    """
    b, s, t = references.shape
    mix = mixture[..., :t].expand(b, s, t)
    if score_fn is si_snr:
        _, perm = si_snr_pit(estimates, references)
    else:
        _, perm = sdr_pit_indices(estimates, references)
    # assign each utterance its best-permutation estimate, broadcast to (B, S, T)
    idx = torch.arange(b, device=estimates.device).unsqueeze(1)
    est_perm = estimates[idx, perm.unsqueeze(1), :].expand(b, s, t)
    base = score_fn(mix, references)          # (B, S)
    est_s = score_fn(est_perm, references)    # (B, S)
    return (est_s - base).mean(dim=-1)        # (B,)


def sdr_pit_indices(estimates: torch.Tensor, references: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b, s, t = estimates.shape
    e = estimates.unsqueeze(2).expand(b, s, s, t).reshape(-1, t)
    r = references.unsqueeze(1).expand(b, s, s, t).reshape(-1, t)
    scores = sdr(e, r).reshape(b, s, s)
    perm0 = scores.diagonal(dim1=1, dim2=2).sum(-1)
    perm1 = scores.flip(dims=(2,)).diagonal(dim1=1, dim2=2).sum(-1)
    return torch.stack([perm0, perm1], dim=-1).max(dim=-1)


def si_snri(estimates: torch.Tensor, references: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
    return _improvement(si_snr, estimates, references, mixture)


def sdri(estimates: torch.Tensor, references: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
    return _improvement(sdr, estimates, references, mixture)


# ---------------------------------------------------------------------------
# Perceptual metrics (computed at 16 kHz on CPU, numpy domain)
# ---------------------------------------------------------------------------
def _to_16k_mono(x: torch.Tensor, sr: int) -> np.ndarray:
    if sr != 16000:
        x = torchaudio_resample(x, sr, 16000)
    return x.squeeze(0).detach().cpu().numpy()


def torchaudio_resample(x: torch.Tensor, src: int, dst: int) -> torch.Tensor:
    import torchaudio.functional as F

    return F.resample(x, src, dst)


def pesq_score(ref: torch.Tensor, est: torch.Tensor, sr: int = 16000) -> float:
    """Wideband PESQ (needs 16 kHz). Returns -0.5 .. 4.5; -1 on failure."""
    try:
        from pesq import pesq as pesq_fn
    except ImportError:
        return float("nan")
    r = _to_16k_mono(ref, sr)
    e = _to_16k_mono(est, sr)
    n = min(r.shape[0], e.shape[0])
    return float(pesq_fn(16000, r[:n], e[:n], "wb"))


def stoi_score(ref: torch.Tensor, est: torch.Tensor, sr: int = 16000) -> float:
    """STOI in [0, 1]."""
    try:
        from pystoi import stoi as stoi_fn
    except ImportError:
        return float("nan")
    r = _to_16k_mono(ref, sr)
    e = _to_16k_mono(est, sr)
    n = min(r.shape[0], e.shape[0])
    return float(stoi_fn(r[:n], e[:n], 16000, extended=False))


def compute_perceptual(est: torch.Tensor, ref: torch.Tensor, sr: int) -> dict:
    """Best-permutation PESQ/STOI for a 2-speaker (1,2,T) estimate."""
    _, perm = si_snr_pit(est, ref)
    perm = perm.item()
    # PESQ/STOI are computed for the PIT-assigned (estimate, reference) pair
    ref_perm = ref[0, perm]
    est_perm = est[0, 0] if perm == 0 else est[0, 1]
    return {
        "pesq": pesq_score(ref_perm, est_perm, sr),
        "stoi": stoi_score(ref_perm, est_perm, sr),
    }


# ---------------------------------------------------------------------------
# Latency helpers
# ---------------------------------------------------------------------------
def measure_latency(model, sample: dict, device: torch.device, repeats: int = 20) -> dict:
    """Wall-clock inference latency (ms) and peak GPU memory (MiB)."""
    model.eval()
    warm = 3
    with torch.no_grad():
        for _ in range(warm):
            model(**sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(**sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
    out = {
        "latency_ms_mean": float(np.mean(times)),
        "latency_ms_std": float(np.std(times)),
        "latency_ms_p95": float(np.percentile(times, 95)),
    }
    if device.type == "cuda":
        out["peak_mem_mib"] = float(torch.cuda.max_memory_allocated() / 2**20)
    return out
