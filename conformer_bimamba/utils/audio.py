"""Audio IO / mixing helpers."""
from __future__ import annotations

import numpy as np
import torch
import torchaudio

EPS = 1e-8


def load_wav(path: str, target_sr: int | None = None) -> torch.Tensor:
    """Load a mono waveform as float32 tensor of shape (1, T).

    Resamples to ``target_sr`` when given. Mirrors the source-rate
    convention of Vivos / Common Voice (both 16 kHz by default).
    """
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:  # downmix to mono
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav - wav.mean()
    if target_sr is not None and sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.float()


def rms(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(EPS)


def set_rms(x: torch.Tensor, target_rms: float = 0.05) -> torch.Tensor:
    """Normalise ``x`` to a fixed RMS."""
    return x * (target_rms / rms(x))


def mix_at_snr(s1: torch.Tensor, s2: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Mix two equal-length signals at a given SNR (s1 is the reference)."""
    scale = rms(s1) / rms(s2).clamp_min(EPS) * (10.0 ** (-snr_db / 20.0))
    return s1 + scale * s2


def make_mixture(
    s1: torch.Tensor,
    s2: torch.Tensor,
    snr_db: float,
    target_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align, normalise and mix two sources -> (mixture, s1, s2).

    Both sources are zero-padded/truncated to ``target_len`` samples.
    Returns tensors of shape (1, target_len).
    """
    s1 = _fit_len(s1, target_len)
    s2 = _fit_len(s2, target_len)
    s1 = set_rms(s1)
    s2 = set_rms(s2)
    mix = mix_at_snr(s1, s2, snr_db)
    return mix, s1, s2


def _fit_len(x: torch.Tensor, n: int) -> torch.Tensor:
    if x.shape[-1] < n:
        pad = n - x.shape[-1]
        x = torch.nn.functional.pad(x, (0, pad))
    elif x.shape[-1] > n:
        x = x[..., :n]
    return x


def resample_tensor(x: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    """Resample a (1, T) or (B, T) waveform tensor."""
    if src_sr == dst_sr:
        return x
    return torchaudio.functional.resample(x, src_sr, dst_sr)


def write_wav(path: str, wav: torch.Tensor, sr: int) -> None:
    torchaudio.save(path, wav.detach().cpu(), sr)


def snr_db_between(mix: torch.Tensor, ref: torch.Tensor) -> float:
    """True SNR (dB) of ``ref`` inside ``mix`` — used for logging."""
    noise = mix - ref
    return 10.0 * torch.log10((ref.pow(2).sum() + EPS) / (noise.pow(2).sum() + EPS)).item()
