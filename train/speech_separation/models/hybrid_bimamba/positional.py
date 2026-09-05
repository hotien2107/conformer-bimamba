"""Sinusoidal positional encoding (Vaswani et al., 2017)."""
from __future__ import annotations

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal PE, added to the chunk/segment dimension.

    Mamba is order-sensitive through its recurrent scan but has no explicit
    position mechanism; Sepformer-style dual-path models add sinusoidal PE to
    both the intra-chunk (S) and inter-chunk (C) dimensions. Position
    information is therefore available to the Bi-Mamba layers.
    """

    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add PE along dim 1. ``x``: (..., L, d_model)."""
        return x + self.pe[:, : x.shape[1]]
