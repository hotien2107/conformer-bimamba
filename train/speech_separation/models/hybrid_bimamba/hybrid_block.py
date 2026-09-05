"""Conformer blocks: baseline (MHSA) and proposed hybrid (Bi-Mamba).

Block layout (identical skeleton for a fair ablation):

    Baseline:  x -> 0.5*FFN -> +MHSA -> +DWConv -> 0.5*FFN -> x
    Hybrid:    x -> 0.5*FFN -> +BiMamba -> +DWConv -> 0.5*FFN -> x   (MHSA removed)

Every sub-layer is wrapped in PreNorm (LayerNorm before the function) and
keeps a residual connection, following the original Conformer (Gulati et
al., 2020). The Depthwise-Conv sub-layer and both FFN sub-layers are
byte-for-byte identical between the two blocks, so any difference in
performance is attributable to the MHSA -> Bi-Mamba swap.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .bimamba import BiMambaLayer


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), *args, **kwargs)


class FeedForward(nn.Module):
    """Conv-FFN (kernel=1 -> pure MLP), expansion factor 4 (Conformer default)."""

    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DepthwiseConv1d(nn.Module):
    """Depthwise separable conv (kernel ~31 @ 8 kHz encoder rate < 4 ms... at
    32 kHz encoder stride 8 -> 250 us/sample; 31 taps ~ 7.75 ms of context —
    local phoneme/plosive features). 'same' padding, SiLU, BatchNorm."""

    def __init__(self, d_model: int, kernel: int = 31, dropout: float = 0.0):
        super().__init__()
        assert kernel % 2 == 1, "odd kernel for 'same' padding"
        self.conv = nn.Conv1d(d_model, d_model, kernel, groups=d_model, padding=kernel // 2, bias=True)
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, D) -> (B, L, D)."""
        x = x.transpose(1, 2)
        x = self.bn(self.conv(x))
        x = self.act(x)
        x = self.dropout(x)
        return x.transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x, need_weights=False)
        return out


class ConformerBlock(nn.Module):
    """Baseline Conformer block: FFN -> MHSA -> DWConv -> FFN."""

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        d_ffn: int = 1024,
        conv_kernel: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ffn1 = PreNorm(d_model, FeedForward(d_model, d_ffn, dropout))
        self.attn = PreNorm(d_model, MultiHeadSelfAttention(d_model, nhead, dropout))
        self.conv = PreNorm(d_model, DepthwiseConv1d(d_model, conv_kernel))
        self.ffn2 = PreNorm(d_model, FeedForward(d_model, d_ffn, dropout))

    def forward(self, x: torch.Tensor, f0: torch.Tensor | None = None) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        x = x + self.attn(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return x


class HybridBlock(nn.Module):
    """Proposed hybrid block: FFN -> Bi-Mamba -> DWConv -> FFN.

    The MHSA sub-layer is *removed* and replaced by BiMambaLayer; the
    Depthwise-Conv sub-layer (local features) and both FFN sub-layers are
    untouched. The Bi-Mamba output is added to its input (residual) and
    normalised (via the next PreNorm) before the Depthwise-Conv layer —
    exactly the Conformer residual/norm contract.
    """

    def __init__(
        self,
        d_model: int,
        d_ffn: int = 1024,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        bimamba_kwargs: dict | None = None,
    ):
        super().__init__()
        bimamba_kwargs = dict(bimamba_kwargs or {})
        self.ffn1 = PreNorm(d_model, FeedForward(d_model, d_ffn, dropout))
        self.mamba = PreNorm(d_model, BiMambaLayer(d_model, **bimamba_kwargs))
        self.conv = PreNorm(d_model, DepthwiseConv1d(d_model, conv_kernel))
        self.ffn2 = PreNorm(d_model, FeedForward(d_model, d_ffn, dropout))

    def forward(self, x: torch.Tensor, f0: torch.Tensor | None = None) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        x = x + self.mamba(x, f0)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return x
