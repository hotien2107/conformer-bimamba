"""
Hybrid Conformer-BiMamba speech separation model, packaged in the same
style as ClearerVoice-Studio / MossFormer2_SS (modelscope/ClearerVoice-Studio).

Differences from the MOSSFormer2_SS packaging:
  * the global self-attention / FSMN sequence model is replaced by our
    dual-path stack of **Hybrid blocks** (FFN -> Bi-Mamba -> DepthwiseConv
    -> FFN), where Multi-Head Self-Attention is replaced by a bidirectional
    Mamba layer;
  * optional **F0 (tone) conditioning** for Vietnamese: the mixture F0
    contour modulates the Mamba B, C input projections (use_f0=1).

Interface (identical to models/mossformer2/mossformer2.py):
    net = HybridBiMamba_SS(args).model        # .model attribute
    out = net(mixture)                        # mixture: (B, T)
    # -> list of num_spks tensors, each (B, T) waveform (trimmed to input T)
    out = net(mixture, f0)                    # f0: (B, Lf) Hz @ 100 fps (use_f0=1)

The core modules (ssm.py / bimamba.py / hybrid_block.py / f0_conditioning.py /
positional.py) are shared with the standalone project implementation and were
validated with forward/backward smoke tests.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hybrid_bimamba.f0_conditioning import F0Projector, resample_f0
from models.hybrid_bimamba.hybrid_block import ConformerBlock, HybridBlock
from models.hybrid_bimamba.positional import SinusoidalPositionalEncoding

EPS = 1e-8

# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    """Conv1d encoder: (B, T) waveform -> (B, N, T') features.

    Same role as models/mossformer2/mossformer2.py::Encoder (kernel = 16,
    stride = kernel // 2), with the input unsqueezed to a single channel.
    """

    def __init__(self, kernel_size=16, stride=8, out_channels=256):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(1, out_channels, kernel_size, stride, bias=False)
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        """x: (B, T) -> (B, N, T')"""
        x = torch.unsqueeze(x, dim=1)
        x = self.conv1d(x)
        x = F.relu(x)
        # global normalisation of the embedding (Sepformer-style)
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
        return x


class Decoder(nn.Module):
    """ConvTranspose1d decoder: (B, N, T') -> (B, T_dec)."""

    def __init__(self, kernel_size=16, stride=8, in_channels=256):
        super(Decoder, self).__init__()
        self.convtr = nn.ConvTranspose1d(
            in_channels, 1, kernel_size, stride, bias=False
        )
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        """x: (B, N, T') -> (B, T_dec)"""
        x = self.convtr(x)
        return torch.squeeze(x, dim=1)


class DualPathProcessor(nn.Module):
    """Chunk -> intra hybrid stack -> inter hybrid stack -> overlap-add.

    (adapted from the standalone project; the dual-path skeleton matches
    Sepformer/DPRNN and is what MOSSFormer2 also uses for its MaskNet.)
    """

    def __init__(
        self,
        d_model=256,
        chunk_size=250,
        hop=125,
        block_type="hybrid",
        num_intra=4,
        num_inter=4,
        d_ffn=1024,
        conv_kernel=31,
        nhead=8,
        dropout=0.1,
        f0_dim=64,
        use_f0=True,
        bimamba_kwargs=None,
        checkpointing=False,
    ):
        super(DualPathProcessor, self).__init__()
        assert chunk_size == 2 * hop, "triangular overlap-add assumes chunk_size == 2*hop"
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

        def make_block(i):
            if block_type == "hybrid":
                return HybridBlock(
                    d_model, d_ffn=d_ffn, conv_kernel=conv_kernel,
                    dropout=dropout, bimamba_kwargs=bimamba_kwargs,
                )
            return ConformerBlock(
                d_model, nhead=nhead, d_ffn=d_ffn,
                conv_kernel=conv_kernel, dropout=dropout,
            )

        self.intra_blocks = nn.ModuleList([make_block(i) for i in range(num_intra)])
        self.inter_blocks = nn.ModuleList([make_block(i) for i in range(num_inter)])

    # ------------------------------------------------------------------
    def _apply_blocks(self, blocks, x, f0):
        for blk in blocks:
            if self.checkpointing:
                from torch.utils.checkpoint import checkpoint

                x = checkpoint(blk, x, f0, use_reentrant=False) if f0 is not None else checkpoint(
                    blk, x, use_reentrant=False
                )
            else:
                x = blk(x, f0) if f0 is not None else blk(x)
        return x

    def forward(self, x, f0_emb=None):
        """x: (B, N, T'), f0_emb: (B, T', f0_dim) or None -> (B, N, T')"""
        B, N, T = x.shape
        pad = (self.hop - (T - self.chunk_size) % self.hop) % self.hop
        if pad > 0:
            x = F.pad(x, (0, pad))
        T_pad = x.shape[-1]

        chunks = x.unfold(-1, self.chunk_size, self.hop)  # (B, N, C, S)
        chunks = chunks.permute(0, 2, 3, 1)  # (B, C, S, N)
        B_, C, S, _ = chunks.shape

        # --- F0 chunks (exact per-position conditioning for the intra path) ---
        f0_chunks = None
        f0_stats = None
        if f0_emb is not None:
            f0_emb_p = F.pad(f0_emb, (0, 0, 0, pad)) if pad else f0_emb
            f0_chunks = f0_emb_p.unfold(1, self.chunk_size, self.hop).permute(0, 1, 3, 2)
            f0_stats = f0_chunks.mean(dim=2)  # (B, C, f0_dim)

        # --- positional encoding + intra-chunk processing ---
        chunks = chunks + self.pe_intra.pe[:, :S].unsqueeze(1)
        h = chunks.reshape(B * C, S, N)
        h_f0 = None if f0_chunks is None else f0_chunks.reshape(B * C, S, self.f0_dim)
        h = self._apply_blocks(self.intra_blocks, h, h_f0)
        chunks = h.view(B, C, S, N)

        # --- inter-chunk processing (per-chunk F0 summary) ---
        chunks = chunks.transpose(1, 2)  # (B, S, C, N)
        chunks = chunks + self.pe_inter.pe[:, :C].unsqueeze(0)
        h = chunks.reshape(B * S, C, N)
        if f0_stats is not None:
            h_f0 = f0_stats.unsqueeze(1).expand(B, S, C, self.f0_dim).reshape(
                B * S, C, self.f0_dim
            )
        else:
            h_f0 = None
        h = self._apply_blocks(self.inter_blocks, h, h_f0)
        chunks = h.view(B, S, C, N).transpose(1, 2)  # (B, C, S, N)

        # --- overlap-add with a triangular window (sums to 1, S == 2*hop) ---
        window = torch.cat(
            [torch.linspace(0, 1, self.hop), torch.linspace(1, 0, self.hop)], dim=0
        ).to(chunks)
        chunks = chunks * window.view(1, 1, S, 1)
        out_T = (C - 1) * self.hop + S
        out = chunks.new_zeros(B, out_T, N)
        for c in range(C):
            out[:, c * self.hop : c * self.hop + S] += chunks[:, c]
        return out.transpose(1, 2)[..., :T]  # (B, N, T)


class HybridBiMamba(nn.Module):
    """Time-domain encoder -> dual-path hybrid stack -> mask decoder.

    Arguments mirror models/mossformer2/mossformer2.py::MossFormer:
        in_channels  : encoder feature width (encoder_embedding_dim)
        out_channels : dual-path model width (mossformer_sequence_dim)
        num_blocks   : number of hybrid blocks per path
        kernel_size  : encoder/decoder kernel
        num_spks     : number of speakers (sources)
        ...
    """

    def __init__(
        self,
        in_channels=256,
        out_channels=256,
        num_intra=4,
        num_inter=4,
        kernel_size=16,
        num_spks=2,
        chunk_size=250,
        hop=125,
        d_ffn=1024,
        conv_kernel=31,
        nhead=8,
        dropout=0.1,
        d_state=64,
        d_conv=4,
        expand=2,
        combine="sum",
        use_mamba2=False,
        use_f0=True,
        f0_dim=64,
        checkpointing=False,
    ):
        super(HybridBiMamba, self).__init__()
        self.num_spks = num_spks
        self.use_f0 = use_f0
        self.f0_dim = f0_dim

        self.enc = Encoder(kernel_size=kernel_size, stride=kernel_size // 2, out_channels=in_channels)

        if use_f0:
            self.f0_proj = F0Projector(f0_dim=f0_dim)

        bimamba_kwargs = dict(
            combine=combine,
            f0_dim=f0_dim if use_f0 else 0,
            use_mamba2=use_mamba2,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dual_path = DualPathProcessor(
            d_model=out_channels,
            chunk_size=chunk_size,
            hop=hop,
            block_type="hybrid",
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

        # masks per speaker (gated tanh*sigmoid, as in MossFormer2's MaskNet)
        self.conv1d_out = nn.Conv1d(out_channels, out_channels * num_spks, kernel_size=1)
        self.prelu = nn.PReLU()
        self.output = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1), nn.Tanh())
        self.output_gate = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid())
        self.dec = Decoder(kernel_size=kernel_size, stride=kernel_size // 2, in_channels=out_channels)

    def forward(self, input, f0=None):
        """input: (B, T); f0: (B, Lf) Hz @ 100 fps or None.
        Returns a list of num_spks tensors, each (B, T) trimmed to input T."""
        x = self.enc(input)  # (B, N, T')
        T_enc = x.shape[-1]

        f0_emb = None
        if f0 is not None and self.use_f0:
            f0_r = resample_f0(f0, f0.shape[-1], T_enc)  # (B, T_enc)
            f0_emb = self.f0_proj(f0_r)  # (B, T_enc, f0_dim)

        h = self.dual_path(x, f0_emb)  # (B, N, T_enc)
        h = self.prelu(h)

        masks = self.conv1d_out(h)  # (B, num_spks * N, T_enc)
        B, _, T_e = masks.shape
        masks = masks.view(B * self.num_spks, -1, T_e)
        masks = self.output(masks) * self.output_gate(masks)

        # apply masks to the encoder embedding and decode each speaker
        x_stacked = torch.stack([x] * self.num_spks).view(B * self.num_spks, -1, T_enc)
        sep = x_stacked * masks
        est = self.dec(sep).view(B, self.num_spks, -1)  # (B, num_spks, T_dec)

        T_origin = input.size(1)
        T_est = est.size(2)
        if T_origin > T_est:
            est = F.pad(est, (0, T_origin - T_est))
        else:
            est = est[:, :, :T_origin]

        out = [est[:, spk, :] for spk in range(self.num_spks)]
        return out


class HybridBiMamba_SS(nn.Module):
    """HybridBiMamba wrapper for ClearerVoice-Studio (mirrors MossFormer2_SS).

    Reads the standard training/inference config keys and exposes ``.model``.
    """

    def __init__(self, args):
        super(HybridBiMamba_SS, self).__init__()
        self.model = HybridBiMamba(
            in_channels=args.encoder_embedding_dim,
            out_channels=args.encoder_embedding_dim,
            num_intra=args.num_intra,
            num_inter=args.num_inter,
            kernel_size=args.encoder_kernel_size,
            num_spks=args.num_spks,
            chunk_size=args.chunk_size,
            hop=args.chunk_size // 2,
            d_ffn=args.d_ffn,
            conv_kernel=args.conv_kernel,
            nhead=args.nhead,
            dropout=args.dropout,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
            combine=args.combine,
            use_mamba2=args.use_mamba2,
            use_f0=bool(getattr(args, "use_f0", 0)),
            f0_dim=args.f0_dim,
            checkpointing=bool(getattr(args, "checkpointing", 0)),
        )

    def forward(self, x, f0=None):
        return self.model(x, f0)
