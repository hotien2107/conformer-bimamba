"""F0 (pitch) conditioning front-end.

Vietnamese is tonal: the six tones are realised mainly through F0 contour
shapes (level / rising / falling / falling-rising / creaky etc.). We extract
an F0 contour from the *mixture* (offline, see ``preprocess/extract_f0.py``)
and use it to modulate the state-space input projections B, C of every
Bi-Mamba layer — i.e. the selective-scan transition is conditioned on the
instantaneous tonal context of the speakers in the mix.

This module is split in two:

* ``F0Projector``  — model-level: raw per-frame F0 (Hz) -> shared embedding
  (B, T, f0_dim). Also computes the three scalar features per frame:
  log-F0, voiced flag, and log-F0 normalised by the utterance median
  (median normalisation removes the speaker-identity component of pitch,
  keeping the *contour* — the tone carrier).
* the per-layer modulation gate lives inside
  ``models.ssm.F0ConditionedSSM`` (``f0_gate``: f0_dim -> 2*d_state with
  tanh, applied as adaptive scaling ``B * (1 + tanh(gate))``).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class F0Projector(nn.Module):
    """(B, T, 1) raw F0 in Hz -> (B, T, f0_dim) embedding."""

    def __init__(self, f0_dim: int = 64, hidden: int = 32):
        super().__init__()
        self.f0_dim = f0_dim
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, f0_dim),
        )

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        """``f0``: (B, T) raw Hz (0 = unvoiced). Returns (B, T, f0_dim)."""
        B, T = f0.shape
        x = f0.clamp_min(1e-3)
        voiced = (f0 > 1.0).float()  # (B, T)
        log_f0 = torch.log(x)

        # Per-utterance median of voiced F0 (speaker-normalisation).
        medians = []
        for b in range(B):
            vals = x[b][voiced[b] > 0.5]
            medians.append(vals.median() if vals.numel() > 0 else x[b].median())
        med = torch.stack(medians).clamp_min(1e-3).view(B, 1)  # (B, 1)
        norm_log_f0 = log_f0 - torch.log(med)

        feats = torch.stack([log_f0, voiced, norm_log_f0], dim=-1)  # (B, T, 3)
        return self.net(feats)


def resample_f0(
    f0: torch.Tensor | None,
    src_len: int,
    dst_len: int,
    threshold: float = 0.5,
) -> torch.Tensor | None:
    """Resample an F0 contour from ``src_len`` frames to ``dst_len`` frames.

    F0 and its voiced/unvoiced mask are interpolated separately; a frame is
    voiced only if the interpolated mask exceeds ``threshold``, which avoids
    smearing pitch energy into unvoiced regions. Returns (B, dst_len).
    """
    if f0 is None:
        return None
    B = f0.shape[0]
    if src_len == dst_len:
        return f0.float()
    mask = (f0 > 1.0).float().unsqueeze(1)  # (B, 1, src)
    val = f0.float().clamp_min(0.0).unsqueeze(1)
    mask = F.interpolate(mask, size=dst_len, mode="linear", align_corners=False)
    val = F.interpolate(val, size=dst_len, mode="linear", align_corners=False)
    out = val * (mask > threshold).float()
    return out.squeeze(1)  # (B, dst_len)
