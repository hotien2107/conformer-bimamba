"""Hybrid Conformer-BiMamba dual-path separator (Sepformer-style backbone).

Pipeline
--------
mixture (B, 1, T)
  -> Conv1d encoder (kernel=16, stride=8) + global normalisation   (B, N, T')
  -> [optional] F0Projector: raw F0 -> (B, T', f0_dim)
  -> Dual-path processing over chunks:
       intra: hybrid blocks over the chunk dimension (S)
       inter: hybrid blocks over the chunk-index dimension (C)
       (sinusoidal PE added to both dimensions, as in Sepformer)
  -> overlap-add back to (B, N, T')
  -> masker: Linear(N, 2N) -> sigmoid -> 2 masks
  -> ConvTranspose1d decoder -> (B, 2, 1, T_dec)

Both the intra- and inter-chunk stacks use the same block type
(``HybridBlock`` or ``ConformerBlock``), so a single ``block_type`` config
switch reproduces the baseline (vanilla Conformer) and the proposed model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .f0_conditioning import F0Projector, resample_f0
from .hybrid_block import ConformerBlock, HybridBlock
from .positional import SinusoidalPositionalEncoding

EPS = 1e-8


class Encoder(nn.Module):
    """Learned 1D-conv encoder with global-norm (Sepformer-style)."""

    def __init__(self, kernel_size: int = 16, stride: int = 8, out_channels: int = 256):
        super().__init__()
        self.conv = nn.Conv1d(1, out_channels, kernel_size, stride, bias=False)
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, wav: torch.Tensor) -> torch.Tensor:  # (B, 1, T) -> (B, N, T')
        x = self.conv(wav)
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
        return x


class Decoder(nn.Module):
    """Transposed conv back to waveform."""

    def __init__(self, kernel_size: int = 16, stride: int = 8, in_channels: int = 256):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, 1, kernel_size, stride, bias=False)
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, T') -> (B, 1, T_dec)
        return self.conv(x)


class DualPathProcessor(nn.Module):
    """Chunk -> intra stack -> inter stack -> overlap-add."""

    def __init__(
        self,
        d_model: int,
        chunk_size: int = 250,
        hop: int = 125,
        block_type: str = "hybrid",
        num_intra: int = 4,
        num_inter: int = 4,
        d_ffn: int = 1024,
        conv_kernel: int = 31,
        nhead: int = 8,
        dropout: float = 0.1,
        f0_dim: int = 64,
        use_f0: bool = True,
        bimamba_kwargs: dict | None = None,
        checkpointing: bool = False,
    ):
        super().__init__()
        assert chunk_size == 2 * hop, "triangular overlap-add window assumes chunk_size == 2*hop"
        assert block_type in ("hybrid", "conformer")
        if use_f0 and block_type == "conformer":
            raise ValueError("use_f0=True requires block_type='hybrid'")

        self.chunk_size = chunk_size
        self.hop = hop
        self.block_type = block_type
        self.checkpointing = checkpointing
        self.use_f0 = use_f0
        self.f0_dim = f0_dim

        self.pe_intra = SinusoidalPositionalEncoding(d_model, max_len=chunk_size)
        self.pe_inter = SinusoidalPositionalEncoding(d_model, max_len=10000)

        block_fn = (
            (lambda i: HybridBlock(d_model, d_ffn=d_ffn, conv_kernel=conv_kernel,
                                   dropout=dropout, bimamba_kwargs=bimamba_kwargs))
            if block_type == "hybrid"
            else (lambda i: ConformerBlock(d_model, nhead=nhead, d_ffn=d_ffn,
                                           conv_kernel=conv_kernel, dropout=dropout))
        )
        self.intra_blocks = nn.ModuleList([block_fn(i) for i in range(num_intra)])
        self.inter_blocks = nn.ModuleList([block_fn(i) for i in range(num_inter)])

    # ------------------------------------------------------------------
    def _apply_blocks(self, blocks: nn.ModuleList, x: torch.Tensor, f0: torch.Tensor | None):
        for blk in blocks:
            if self.checkpointing:
                x = checkpoint(blk, x, f0, use_reentrant=False) if f0 is not None else checkpoint(
                    blk, x, use_reentrant=False
                )
            else:
                x = blk(x, f0) if f0 is not None else blk(x)
        return x

    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        T = x.shape[-1]
        pad = (self.hop - (T - self.chunk_size) % self.hop) % self.hop
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, pad))
        return x, pad

    def _chunk(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, T_pad) -> (B, C, S, N)."""
        chunks = x.unfold(-1, self.chunk_size, self.hop)  # (B, N, C, S)
        return chunks.permute(0, 2, 3, 1)

    def _overlap_add(self, chunks: torch.Tensor) -> torch.Tensor:
        """(B, C, S, N) -> (B, N, T). Triangular window; sums to 1 since S == 2*hop."""
        B, C, S, N = chunks.shape
        window = torch.cat(
            [torch.linspace(0, 1, self.hop), torch.linspace(1, 0, self.hop)], dim=0
        ).to(chunks)
        chunks = chunks * window.view(1, 1, S, 1)
        T = (C - 1) * self.hop + S
        out = chunks.new_zeros(B, T, N)
        for c in range(C):
            out[:, c * self.hop : c * self.hop + S] += chunks[:, c]
        return out.transpose(1, 2)  # (B, N, T)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, f0_emb: torch.Tensor | None = None) -> torch.Tensor:
        """``x``: (B, N, T'); ``f0_emb``: (B, T', f0_dim) or None -> (B, N, T')."""
        B, N, T = x.shape
        x, pad = self._pad(x)
        T_pad = x.shape[-1]

        chunks = self._chunk(x)  # (B, C, S, N)
        B_, C, S, _ = chunks.shape

        # --- F0 chunks for the intra path: exact per-position conditioning ---
        f0_chunks = None
        f0_stats = None
        if f0_emb is not None:
            f0_emb_p = torch.nn.functional.pad(f0_emb, (0, 0, 0, pad)) if pad else f0_emb
            # unfold appends the new dim: (B, C, f0_dim, S) -> permute to (B, C, S, f0_dim)
            f0_chunks = f0_emb_p.unfold(1, self.chunk_size, self.hop).permute(0, 1, 3, 2)
            # per-chunk F0 summary (mean over S) used to condition the inter path
            f0_stats = f0_chunks.mean(dim=2)  # (B, C, f0_dim)

        # --- positional encoding ---
        chunks = chunks + self.pe_intra.pe[:, :S].unsqueeze(1)  # (B, C, S, N)

        # --- intra-chunk processing (over the S dimension) ---
        h = chunks.reshape(B * C, S, N)
        h_f0 = None if f0_chunks is None else f0_chunks.reshape(B * C, S, self.f0_dim)
        h = self._apply_blocks(self.intra_blocks, h, h_f0)
        chunks = h.view(B, C, S, N)

        # --- inter-chunk processing (over the C dimension) ---
        chunks = chunks.transpose(1, 2)  # (B, S, C, N)
        chunks = chunks + self.pe_inter.pe[:, :C].unsqueeze(0)
        h = chunks.reshape(B * S, C, N)
        if f0_stats is not None:
            # per-chunk F0 summary broadcast across positions inside a chunk
            h_f0 = f0_stats.unsqueeze(1).expand(B, S, C, self.f0_dim).reshape(B * S, C, self.f0_dim)
        else:
            h_f0 = None
        h = self._apply_blocks(self.inter_blocks, h, h_f0)
        chunks = h.view(B, S, C, N).transpose(1, 2)  # (B, C, S, N)

        out = self._overlap_add(chunks)  # (B, N, T_pad)
        return out[..., :T]


class HybridSepformer(nn.Module):
    """Full separator. ``block_type``: 'hybrid' (proposed) or 'conformer' (baseline)."""

    def __init__(
        self,
        encoder_dim: int = 256,
        encoder_kernel: int = 16,
        encoder_stride: int = 8,
        chunk_size: int = 250,
        hop: int = 125,
        block_type: str = "hybrid",
        num_intra: int = 4,
        num_inter: int = 4,
        d_ffn: int = 1024,
        conv_kernel: int = 31,
        nhead: int = 8,
        dropout: float = 0.1,
        use_f0: bool = True,
        f0_dim: int = 64,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        combine: str = "sum",
        use_mamba2: bool = False,
        checkpointing: bool = False,
    ):
        super().__init__()
        if use_f0 and use_mamba2:
            raise ValueError("use_mamba2 (fused Mamba-2) is incompatible with F0 conditioning")
        if use_f0 and block_type != "hybrid":
            raise ValueError("F0 conditioning only applies to the hybrid block")

        self.encoder = Encoder(encoder_kernel, encoder_stride, encoder_dim)
        self.decoder = Decoder(encoder_kernel, encoder_stride, encoder_dim)

        bimamba_kwargs = dict(
            combine=combine,
            f0_dim=f0_dim if use_f0 else 0,
            use_mamba2=use_mamba2,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dual_path = DualPathProcessor(
            d_model=encoder_dim,
            chunk_size=chunk_size,
            hop=hop,
            block_type=block_type,
            num_intra=num_intra,
            num_inter=num_inter,
            d_ffn=d_ffn,
            conv_kernel=conv_kernel,
            nhead=nhead,
            dropout=dropout,
            f0_dim=f0_dim,
            use_f0=use_f0,
            bimamba_kwargs=bimamba_kwargs,
            checkpointing=checkpointing,
        )
        self.masker = nn.Linear(encoder_dim, 2 * encoder_dim)
        self.use_f0 = use_f0
        self.f0_dim = f0_dim
        if use_f0:
            self.f0_proj = F0Projector(f0_dim=f0_dim)

    def forward(
        self,
        mixture: torch.Tensor,
        f0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``mixture``: (B, 1, T); ``f0``: (B, Lf) raw Hz or None.
        Returns estimates (B, 2, 1, T_dec)."""
        e = self.encoder(mixture)  # (B, N, T')
        T_enc = e.shape[-1]

        f0_emb = None
        if f0 is not None and self.use_f0:
            f0_r = resample_f0(f0, f0.shape[-1], T_enc)  # (B, T_enc)
            f0_emb = self.f0_proj(f0_r)  # (B, T_enc, f0_dim)

        h = self.dual_path(e, f0_emb)  # (B, N, T_enc)

        m = self.masker(h.transpose(1, 2))  # (B, T_enc, 2N)
        m1, m2 = m.chunk(2, dim=-1)
        m1, m2 = torch.sigmoid(m1), torch.sigmoid(m2)
        e_t = e.transpose(1, 2)  # (B, T_enc, N)
        est1 = self.decoder((m1 * e_t).transpose(1, 2))
        est2 = self.decoder((m2 * e_t).transpose(1, 2))
        return torch.stack([est1, est2], dim=1)  # (B, 2, 1, T_dec)

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if not trainable_only or p.requires_grad)
