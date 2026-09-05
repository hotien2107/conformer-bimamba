# Disadvantages & Limitations of the Hybrid Conformer-BiMamba Model

*An honest, critical assessment. Every number below was measured by
instantiating the actual models in this repository at the default config
(`configs/*.yaml`).*

---

## 0. Summary (TL;DR)

| # | Disadvantage | Severity | Mitigation |
|---|---|---|---|
| D1 | **More parameters than the MHSA baseline** (+53%: 16.4 M vs 10.7 M) — the O(n) claim is about *scaling with sequence length*, not model size | High | `expand=1`, `d_state=32`, fewer blocks, or lower `d_model` |
| D2 | Mamba training is LR-sensitive / can be unstable | High | `mamba_lr_scale=0.5`, 5 k warmup, grad clip, conservative `dt` init |
| D3 | Bidirectional = 2× single-Mamba cost; gradient checkpointing slows training | Medium | Keep only when memory-bound; `use_mamba2` fast path |
| D4 | mamba-ssm is Linux+CUDA-only; no fast CPU kernel; edge deployment needs custom ports | High | Pure-PyTorch fallback (dev only); C++/ONNX port for production |
| D5 | F0 conditioning is only as good as the pitch tracker; mixture F0 is ill-defined for 2 voices | Medium | Median normalisation, stonemask smoothing, soft tanh gate, no-F0 ablation |
| D6 | Non-causal by design — not usable for streaming/online separation | Medium | Documented; causal variant = half the model |
| D7 | Dual-path chunking (250/125) bounds the attention quadratic anyway → the O(n) win is real but smaller than full-sequence comparisons suggest | Low | Measure latency explicitly |
| D8 | Vivos is small; SI-SNR does not measure tonal correctness | Medium | Common Voice augmentation; optional ASR probe |

---

## D1. The model is *bigger* than the baseline — the complexity argument is about scaling, not size

Measured parameter counts (default config: `d_model=256`, `d_ffn=1024`, 4+4 blocks):

| Model | Parameters | vs baseline |
|---|---|---|
| Vanilla Conformer (MHSA) | **10,740,224 (10.74 M)** | — |
| Hybrid Bi-Mamba (no F0) | **16,294,400 (16.29 M)** | **+5.55 M (+51.7 %)** |
| Hybrid Bi-Mamba + F0 | **16,430,816 (16.43 M)** | **+5.69 M (+53.0 %)** |

Why: per block, the MHSA sub-layer costs 263,680 params (4·d² ≈ 0.26 M) while the
Bi-Mamba sub-layer costs **974,592** (two cores × ~487 k; each core = `in_proj`
262 k + `out_proj` 131 k + `x_proj` 74 k + `dt_proj` 8.7 k + `f0_gate` 8.3 k +
`conv1d` 2.6 k + `D` 0.5 k). **Bi-Mamba ≈ 3.7× the attention sub-layer**; the
two directions make it ~7.4× the attention cost in that slot. The FFNs
(1.05 M/block) dominate both models equally, so the swap adds +710,912 per
block × 8 blocks ≈ +5.7 M.

Consequences:
- Higher VRAM for the *same width* and more FLOPs per step in the projections
  (the saving is on the attention matrix, which the scan replaces).
- The honest framing: Mamba's advantage is **O(n) compute/memory vs O(n²) for
  the attention matrix** at growing sequence length — *not* a smaller network.
- To reach parameter parity with the baseline one must shrink the hybrid
  (`expand=1`, `d_state=32`, or fewer hybrid blocks), which may cost accuracy.
- Literature hybrid SSM/Transformer models (e.g., MOSSFormer for separation,
  Samba for LM) typically use *smaller* d_model for the Mamba branch
  precisely for this reason.

## D2. Training stability of Mamba

- SSM layers are known to be sensitive to learning rate: `dt` and `A` live on
  a log-scale; too-large LR pushes `softplus(dt + dt_bias)` out of range and
  the scan becomes explosive or dead.
- Mitigations implemented (and they are *costs*, not free): per-parameter-group
  LR 0.5× for SSM params, 5,000-step warmup, grad clip 5.0, conservative
  `dt_proj` initialisation. These slow early convergence relative to the
  baseline.
- fp16 AMP interacts badly with tiny `dt` values (underflow) — the CUDA kernel
  mitigates, but any custom/CPU path must keep `dt` in fp32.

## D3. Bidirectionality doubles the Mamba cost

- Two cores (fwd + bwd) = 2× params, 2× scan time, 2× activation memory vs a
  single-direction Mamba. Still cheaper than attention for long sequences, but
  the factor is real.
- Gradient checkpointing (`checkpointing: true`) reduces activation memory but
  adds recompute overhead (typically +20–40 % step time).

## D4. Dependency and deployment friction

- `mamba-ssm` builds only on Linux with a CUDA toolchain (C++/CUDA/ninja);
  version coupling with `causal-conv1d` and Triton is common pain.
- **No fast CPU kernel exists**: the pure-PyTorch reference scan is sequential
  over time and unsuitable for production. Measured on this Mac (Apple M-series,
  toy model 24-dim/2 blocks): ≈ 158 ms per 1 s of audio — i.e. slower than
  real-time by ~6× for a *tiny* model; a full 256-dim model would be far
  slower. Edge/CPU deployment requires a C++/ONNX/GGML port of the scan.
- `Mamba2`'s fused kernel cannot accept external B, C, so the F0-conditioned
  path is tied to the older Mamba-1 kernel (slower, extra memory).
- On non-CUDA boxes the whole model silently runs the reference scan: correct
  but slow — a reproducibility trap for teams without the exact GPU stack.

## D5. F0 conditioning weaknesses

1. **Mixture F0 is not well defined.** Two simultaneous speakers → two pitch
   tracks; PyWorld returns one. The training proxy `max(f1, f2)` (dominant
   pitch) is a crude approximation of what the model sees at inference
   (PyWorld on the real mixture) — a **train/inference distribution shift**.
2. **Pitch tracker errors**: octave errors, unvoiced frames, and creaky
   phonation (Vietnamese thanh ngã / thanh nặng are often creaky → weak F0)
   corrupt the conditioning signal.
3. **Extra pipeline & latency**: at inference the model needs F0 of the
   mixture *before* separation → an extra PyWorld pass (~5–10× real time on
   CPU) or a learned F0 estimator.
4. **The gate may be ignored.** The gate is a tanh-modulated scaling on B, C;
   if the model learns `gate ≈ 0`, conditioning is silently inert — ablation 2
   vs 3 must be checked (if SI-SNRi is equal, the gate collapsed).
5. **Limited capacity**: `f0_dim=64 → 2·d_state=128` scalars per layer is a
   coarse channel for tonal information (only 3 scalar features per frame).
6. **Speaker-identity shortcut risk**: F0 correlates with speaker identity;
   median-normalisation removes the absolute level, but residual leakage could
   let the model exploit speaker *type* rather than tonal *contour*.

## D6. Non-causal by design

- Bi-Mamba sees the whole utterance — fine for offline separation (WSJ0-2mix
  style) but **not streaming**: a causal variant would keep only the forward
  core, halving the model and losing the tonal *context from both sides*
  (tone contours are often realised over the following syllable — this is
  exactly the information a causal system loses).

## D7. The complexity win is bounded by dual-path chunking

- Attention is applied over chunk dims S=250 and C≈159, **not** the full
  20 k-frame sequence. So the attention matrices are (250×250) and (159×159)
  per head — quadratic but bounded (~318 M floats for intra at B=4, fp32).
  Mamba still wins (state ≈ 83 MB vs attention ≈ 1.3 GB at B=4) but the
  headline "O(n) vs O(n²)" is only fully realised when chunk size grows.
- Overlap-add windowing (hop = chunk/2 constraint) can smear sharp onsets;
  reconstruction quality depends on the triangular window.
- Extra hyperparameters to tune (chunk size, hop, windowing) that the baseline
  also has, but which interact with scan efficiency.

## D8. Data, evaluation, and significance

- **Vivos is small** (~15 h train). Even with Common Voice + dynamic mixing,
  the model may overfit to the ~65 training speakers; SI-SNRi differences of
  ±0.5 dB between ablations may not reach significance — need paired tests
  across many held-out mixtures.
- **SI-SNRi does not measure tonal correctness**: two outputs can score equal
  SI-SNRi while one preserves tone and the other does not. PESQ/STOI help but
  are not tone-aware; an ASR probe on the separated streams is the real test.
- Benchmark comparisons (Sepformer/TF-GridNet) are trained on different data
  (WSJ0-2mix, 8 kHz, 30 h) — numbers are not directly transferable to 32 kHz
  Vietnamese mixtures.
- F0 gains (if any) are language-specific: the *Mamba swap* transfers to any
  language; the *F0 conditioning* is tuned for tonal Vietnamese and may not
  transfer to non-tonal languages.

---

## Suggested reading order
1. [`parameter_analysis.md`](parameter_analysis.md) — exact parameter maths and
   expected-result predictions.
2. [`report.md`](report.md) — architecture details (Vietnamese).
3. [`README.md`](../README.md) — usage, configs, risk table.
