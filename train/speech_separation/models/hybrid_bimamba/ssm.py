"""Selective state-space core with optional F0 conditioning of B, C.

Execution paths
---------------
1. **CUDA (training on GPU)**: uses the Mamba-1 style fused kernel
   ``mamba_ssm.ops.selective_scan_interface.selective_scan_fn`` (stable
   public API across mamba-ssm 1.x and 2.x). The kernel accepts *explicit*
   B and C tensors, which is exactly the injection point we need for F0
   conditioning — B and C are computed from the input and then modulated by
   a per-layer gate driven by the mixture F0 embedding.

2. **Reference (dev / edge / CI)**: a pure-PyTorch sequential scan with the
   same semantics. Slow (sequential over time) but dependency-free; used
   automatically when mamba-ssm is not installed (e.g. macOS, CPU boxes) and
   for small CPU inference demos.

State-space core (Mamba-1 / S6 formulation, Gu & Dao 2023)
----------------------------------------------------------
    dt   = softplus(dt_proj(x) + dt_bias)            # (B, d_inner, L)
    A    = -exp(A_log)                               # (d_inner, d_state)
    x    = silu(depthwise_conv1d(x))                 # local mixing, d_conv=4
    B    = B_proj(x) * (1 + tanh(gate_B(f0)))        # <- F0 modulation
    C    = C_proj(x) * (1 + tanh(gate_C(f0)))        # <- F0 modulation
    y    = selective_scan(x, dt, A, B, C, D)
    y    = y * silu(z)                               # gated output
    out  = out_proj(y)

When ``f0_dim == 0`` the gates are identity, i.e. the layer degenerates to a
plain (single-direction) Mamba block — the "without F0 conditioning" ablation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Fused-kernel availability (mamba-ssm, Linux + CUDA only)
# ---------------------------------------------------------------------------
try:  # mamba-ssm 1.x / 2.x public interface
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _selective_scan_fn

    CUDA_SCAN_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only on non-CUDA hosts
    _selective_scan_fn = None
    CUDA_SCAN_AVAILABLE = False


def selective_scan_ref(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D_skip: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
):
    """Pure-PyTorch reference with the same interface as
    ``mamba_ssm.ops.selective_scan_interface.selective_scan_fn``.

    Shapes: u (B, D, L), delta (B, D, L), A (D, N), B (B, N, L), C (B, N, L),
    D_skip (D,), z (B, D, L). Returns (B, D, L) [and optionally (B, D, N)].
    """
    B_, D, L = u.shape
    N = A.shape[-1]
    if delta_bias is not None:
        delta = delta + delta_bias.view(1, D, 1)
    if delta_softplus:
        delta = F.softplus(delta)

    dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(-2))  # (B, D, L, N)
    dB = delta.unsqueeze(-1) * B.permute(0, 2, 1).unsqueeze(1)          # (B, D, L, N)
    C_t = C.permute(0, 2, 1).unsqueeze(1)                               # (B, 1, L, N)
    u_t = u.unsqueeze(-1)                                               # (B, D, L, 1)

    h = torch.zeros(B_, D, N, device=u.device, dtype=u.dtype)
    ys: list[torch.Tensor] = []
    for t in range(L):  # sequential scan (reference only)
        h = dA[:, :, t] * h + dB[:, :, t] * u_t[:, :, t]
        ys.append((h * C_t[:, :, t]).sum(-1))
    y = torch.stack(ys, dim=-1)  # (B, D, L)
    if D_skip is not None:
        y = y + D_skip.view(1, D, 1) * u
    if z is not None:
        y = y * F.silu(z)
    return (y, h) if return_last_state else y


class F0ConditionedSSM(nn.Module):
    """Single-direction selective SSM (Mamba-1 core) with F0 gating of B, C.

    Args:
        d_model: input/output width.
        d_state: SSM state width N.
        d_conv: depthwise causal conv width before the scan.
        expand: inner expansion factor (d_inner = expand * d_model).
        dt_rank: rank of the dt projection ("auto" -> d_model // 16).
        f0_dim: width of the per-frame F0 embedding (0 disables gating).
        f0_gate_mode: "scale" -> B * (1 + tanh(gate)); "add" -> B + gate.
        use_cuda_scan: prefer the mamba-ssm fused kernel when available.
        dt_min / dt_max: initial dt bias bounds.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str | int = "auto",
        f0_dim: int = 0,
        f0_gate_mode: str = "scale",
        use_cuda_scan: bool = True,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        layer_idx: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.f0_dim = f0_dim
        self.f0_gate_mode = f0_gate_mode
        self.use_cuda_scan = use_cuda_scan and CUDA_SCAN_AVAILABLE
        self.layer_idx = layer_idx

        # Combined input projection: [x, z] only (Mamba-1 layout); dt, B, C
        # are produced by x_proj applied to the conv-activated x (below).
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv, groups=self.d_inner, bias=True, padding=d_conv - 1
        )
        self.act = nn.SiLU()

        # Post-conv projections (B, C are functions of the locally-mixed x)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # --- F0 modulation gate: f0_emb -> 2 * d_state (B-gate, C-gate) ---
        if f0_dim > 0:
            self.f0_gate = nn.Sequential(
                nn.Linear(f0_dim, self.d_state * 2),
                nn.Tanh(),  # values in (-1, 1): stable adaptive scaling
            )
        else:
            self.f0_gate = None

        # SSM parameters (Mamba-1 initialisation)
        A_log = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).unsqueeze(0).repeat(
            self.d_inner, 1
        )
        self.register_buffer("A_log", A_log)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        dt_init_std = self.dt_rank**-0.5
        with torch.no_grad():
            # original Mamba init: small random weight, bias set so that
            # softplus(dt_bias) lands in [dt_min, dt_max]
            self.dt_proj.weight.data.uniform_(-dt_init_std, dt_init_std)
            self.dt_proj.bias.data.copy_(dt)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _modulate(self, B: torch.Tensor, C: torch.Tensor, f0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the F0 gate to B, C. ``B, C``: (B, L, d_state), f0: (B, L, f0_dim)."""
        gate = self.f0_gate(f0)  # (B, L, 2 * d_state)
        gB, gC = gate.chunk(2, dim=-1)
        if self.f0_gate_mode == "scale":
            return B * (1.0 + gB), C * (1.0 + gC)
        if self.f0_gate_mode == "add":
            return B + gB, C + gC
        raise ValueError(f"unknown f0_gate_mode: {self.f0_gate_mode}")

    def forward(self, x: torch.Tensor, f0: torch.Tensor | None = None) -> torch.Tensor:
        """``x``: (B, L, d_model); ``f0``: (B, L, f0_dim) or None -> (B, L, d_model)."""
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_inner, z = xz.split([self.d_inner, self.d_inner], dim=-1)

        x_inner = x_inner.transpose(1, 2)                       # (B, d_inner, L)
        x_inner = self.act(self.conv1d(x_inner)[:, :, :L])      # causal depthwise conv

        x_dbl = self.x_proj(x_inner.transpose(1, 2))            # (B, L, dt_rank + 2*d_state)
        dt, Bmat, Cmat = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.transpose(1, 2)           # (B, d_inner, L)

        if f0 is not None and self.f0_gate is not None:
            Bmat, Cmat = self._modulate(Bmat, Cmat, f0)

        A = -torch.exp(self.A_log)                              # (d_inner, d_state)
        u = x_inner
        B_scan = Bmat.transpose(1, 2)                           # (B, d_state, L)
        C_scan = Cmat.transpose(1, 2)
        z_scan = z.transpose(1, 2)                              # (B, d_inner, L)

        if self.use_cuda_scan and _selective_scan_fn is not None:
            y = _selective_scan_fn(
                u, dt, A, B_scan, C_scan, self.D, z_scan,
                delta_bias=self.dt_proj.bias, delta_softplus=True,
            )
        else:
            y = selective_scan_ref(
                u, dt, A, B_scan, C_scan, self.D, z_scan,
                delta_bias=self.dt_proj.bias, delta_softplus=True,
            )
        y = y.transpose(1, 2)                                   # (B, L, d_inner)
        y = y * self.act(z)
        return self.out_proj(y)


class Mamba2Core(nn.Module):
    """Fused Mamba-2 core (fast path) — no F0 conditioning.

    ``mamba_ssm.Mamba2`` computes B, C internally inside a fused Triton
    kernel, so the F0 injection used by ``F0ConditionedSSM`` is not possible
    without reimplementing the kernel. Use this core only for the
    *unconditioned* ablation (``f0_dim == 0``); it is ~2-3x faster than the
    Mamba-1 kernel and uses less memory.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 256,
        layer_idx: int | None = None,
    ):
        super().__init__()
        try:
            from mamba_ssm import Mamba2
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Mamba2Core requires mamba-ssm (Linux + CUDA). "
                "Use use_mamba2=False to fall back to the F0ConditionedSSM path."
            ) from exc
        self.core = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            chunk_size=chunk_size,
            layer_idx=layer_idx,
        )

    def forward(self, x: torch.Tensor, f0: torch.Tensor | None = None) -> torch.Tensor:
        assert f0 is None, "Mamba2 fused path does not support F0 conditioning"
        return self.core(x)
