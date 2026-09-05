"""Bidirectional Mamba (Bi-Mamba) layer.

Implementation choice
---------------------
**Option A (implemented): two parallel SSM cores.** One core scans the
sequence left-to-right, the other scans it right-to-left (i.e. it consumes
``flip(x)``). The two outputs are combined by summation (default) or
concatenation followed by a linear projection. This makes the layer fully
non-causal — every output position can attend to the whole utterance — which
is exactly what offline speech separation needs (WSJ0-2mix-style evaluation
is non-causal).

**Option B (mamba-ssm built-in bidirectional flag): NOT available.**
Verified against the upstream mamba-ssm 2.x source (May 2025): neither
``Mamba`` nor ``Mamba2`` exposes a ``bidirectional`` argument. If a future
release adds one, it can be switched on here without changing the rest of
the model (see README, "Option B").

No-future-leakage guarantee
---------------------------
Each direction is strictly causal *within its own orientation*: the forward
core at position t depends only on positions <= t; the backward core at
position t depends only on positions >= t. Combining the two streams never
mixes past and future across directions — the model simply has access to the
whole input, which is permitted for offline separation. No autoregressive
(teacher-forcing) style future information is used.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .ssm import F0ConditionedSSM, Mamba2Core


class BiMambaLayer(nn.Module):
    """``FFN -> [this layer] -> DWConv`` replacement for MHSA.

    Args:
        d_model: model width.
        combine: "sum" (default) or "concat" (concat + linear to d_model).
        f0_dim: per-frame F0 embedding width; 0 disables tonal conditioning.
        use_mamba2: use the fused Mamba-2 cores (only valid with f0_dim == 0).
        layer_idx: index used for parameter-naming consistency with mamba-ssm.
        ssm_kwargs: forwarded to the core (d_state, d_conv, expand, ...).
    """

    def __init__(
        self,
        d_model: int,
        combine: str = "sum",
        f0_dim: int = 0,
        use_mamba2: bool = False,
        layer_idx: int | None = None,
        **ssm_kwargs,
    ):
        super().__init__()
        assert combine in ("sum", "concat"), combine
        if use_mamba2 and f0_dim > 0:
            raise ValueError("use_mamba2=True is incompatible with F0 conditioning (f0_dim > 0)")

        core_cls = Mamba2Core if use_mamba2 else F0ConditionedSSM
        core_kwargs = dict(ssm_kwargs)
        if not use_mamba2:
            core_kwargs["f0_dim"] = f0_dim

        self.fwd = core_cls(d_model, layer_idx=layer_idx, **core_kwargs)
        self.bwd = core_cls(d_model, layer_idx=layer_idx, **core_kwargs)
        self.combine = combine
        if combine == "concat":
            self.combine_proj = nn.Linear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor, f0: torch.Tensor | None = None) -> torch.Tensor:
        """``x``: (B, L, d_model); ``f0``: (B, L, f0_dim) or None -> (B, L, d_model)."""
        h_fwd = self.fwd(x, f0)
        x_rev = torch.flip(x, dims=[1])
        f0_rev = None if f0 is None else torch.flip(f0, dims=[1])
        h_bwd = torch.flip(self.bwd(x_rev, f0_rev), dims=[1])

        if self.combine == "sum":
            return h_fwd + h_bwd
        return self.combine_proj(torch.cat([h_fwd, h_bwd], dim=-1))
