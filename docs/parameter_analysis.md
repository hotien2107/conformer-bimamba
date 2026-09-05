# Parameter Analysis & Expected Results

**Hybrid Conformer-BiMamba vs Vanilla Conformer (baseline)**

All counts below are **measured** by instantiating the models in this
repository at the default config (`configs/*.yaml`: `d_model=256`,
`d_ffn=1024`, `conv_kernel=31`, `d_state=64`, `expand=2`, `f0_dim=64`,
4 intra + 4 inter blocks). Counts are trainable parameters unless noted.

---

## 1. Total parameters

| Model | Parameters | vs baseline | vs Sepformer* | 
|---|---|---|---|
| **Vanilla Conformer (MHSA)** — baseline | **10,740,224 (10.74 M)** | — | ~Sepformer-base class |
| **Hybrid Bi-Mamba (no F0)** — ablation 2 | **16,294,400 (16.29 M)** | +5.55 M (+51.7 %) | ~Sepformer-large class |
| **Hybrid Bi-Mamba + F0** — proposed | **16,430,816 (16.43 M)** | +5.69 M (+53.0 %) | ~Sepformer-large class |

\* Literature context: Sepformer-base ≈ 2.6 M, Sepformer-large ≈ 26 M (Subakan
et al., 2021); TF-GridNet ≈ 14 M (Wang et al., 2023). Our hybrid sits between
Sepformer-base and Sepformer-large — same ballpark as TF-GridNet.

**Key takeaway:** the MHSA→Bi-Mamba swap *increases* parameters by ~53 % at the
same width. The O(n) advantage is about **compute/memory scaling with sequence
length**, not about being a smaller network (see `disadvantages.md` D1).

### Cost of the F0 (tonal) conditioning
- Per-layer F0 gates: 8 layers × 2 directions × 8,320 = **133,120**
- Shared `F0Projector`: **3,296**
- **Total F0 machinery: +136,416 params = +0.84 %** over the un-conditioned
  hybrid — essentially free.

---

## 2. Per-component breakdown (proposed model)

| Component | Params | Share | Notes |
|---|---|---|---|
| Encoder (Conv1d 1→256, k16/s8) | 4,096 | 0.02 % | weight only (no bias) |
| Decoder (ConvTranspose1d) | 4,096 | 0.02 % | weight only |
| Masker (Linear 256→512) | 131,584 | 0.8 % | |
| F0Projector (3→32→32→64) | 3,296 | 0.02 % | shared, cheap |
| **Dual-path stack (8 blocks)** | **16,287,744** | **99.1 %** | dominates |
| **Total** | **16,430,816** | 100 % | |

### Per-block: HybridBlock vs ConformerBlock

| Sublayer | HybridBlock | ConformerBlock | Same? |
|---|---|---|---|
| FFN1 + FFN2 (incl. 2× LayerNorm) | 1,052,160 | 1,052,160 | ✅ identical |
| DepthwiseConv1d k31 (incl. LayerNorm, BatchNorm) | 9,216 | 9,216 | ✅ identical |
| **Bi-Mamba sub-layer** (fwd+bwd cores, incl. LayerNorm) | **974,592** | — | MHSA replaced |
| **MHSA sub-layer** (incl. LayerNorm) | — | **263,680** | removed |
| **Block total** | **2,035,968** | **1,325,056** | Δ = +710,912/block |

### Anatomy of one Bi-Mamba core (per direction; ×2 for fwd+bwd)

| Parameter | Shape | Params |
|---|---|---|
| `in_proj` | 256 → 2·512 | 262,144 |
| `out_proj` | 512 → 256 | 131,072 |
| `x_proj` | 512 → dt_rank(16)+2·d_state(128) | 73,728 |
| `dt_proj` | 16 → 512 | 8,704 |
| `f0_gate` | 64 → 2·d_state(128) | 8,320 |
| `conv1d` (d_conv=4, groups=512) | 512×4 + 512 | 2,560 |
| `D` | 512 | 512 |
| **core trainable** | | **487,040** |
| `A_log` (buffer, non-trainable) | 512×64 | 32,768 |

MHSA equivalent: `in_proj` 196,608 + `out_proj` 65,792 + bias 768 ≈ 263 k.

---

## 3. Parameter formulas (any config)

Let `d` = encoder_dim, `f` = d_ffn, `k` = conv_kernel, `d_s` = d_state,
`e` = expand, `r` = dt_rank = ⌈d/16⌉, `f0` = f0_dim, `N` = n_blocks.

```
Encoder   = d · 16
Decoder   = d · 16
Masker    = d · 2d
F0Proj    = 3·32 + 32 + 32·32 + 32 + 32·64 + 64          (= 3,296 for f0=64)

Per ConformerBlock:
  FFN×2    ≈ 2 · (2·d·f)
  MHSA     ≈ 4d² + 4d
  DWConv   ≈ k·d + 5d
Per HybridBlock:
  FFN×2    ≈ 2 · (2·d·f)          (identical)
  BiMamba  ≈ 2 · [ 2·(e·d)·d          (in_proj)
                 + (e·d)·(r + 2·d_s)  (x_proj)
                 + (r+1)·(e·d)        (dt_proj)
                 + (e·d)·d            (out_proj)
                 + (f0+1)·2·d_s       (f0_gate)
                 + 4·(e·d) + (e·d) ]  (conv1d + D)
  DWConv   ≈ k·d + 5d            (identical)
```

---

## 4. Predictions about the output results

These are **falsifiable hypotheses** with ranges based on the literature and
on the analysis above; they must be confirmed by `evaluate.py` runs on the
fixed test set (500 mixtures, 0 dB, 2 speakers, 32 kHz).

### 4.1 Primary metric — SI-SNRi (dB improvement over the mixture)

| Model | Predicted SI-SNRi | Reasoning |
|---|---|---|
| Vanilla Conformer (baseline) | **9.0 – 12.0** | Sepformer-large reaches 20.4 dB on WSJ0-2mix (8 kHz, 0 dB, 30 h training, 26 M params). Our setting is harder per utterance (32 kHz, 4–5 s, Vivos ~15 h + Common Voice) and our model is 10.7 M → expect roughly half the headroom. |
| Hybrid Bi-Mamba (no F0) | **9.5 – 12.5** | Mamba-based separators (e.g., MOSSFormer, ~20.8 dB on WSJ0-2mix at similar size) match Transformer counterparts; expect **+0.0 – +0.8 dB** over the baseline, concentrated on longer utterances where the scan's long-range state pays off. |
| Hybrid Bi-Mamba + F0 (proposed) | **10.0 – 13.5** | Expect **+0.5 – +1.5 dB over the no-F0 hybrid**, *if* the F0 gate activates. Largest gains on **same-gender / pitch-overlapping pairs** and on **tonal words** (contour preserved); near-zero gain on gender-different, well-separated pairs. |

Refined hypothesis: F0 conditioning helps most where attention/Mamba alone
confuse speakers — i.e. pairs whose F0 ranges overlap. A **conditional
analysis** (split test pairs by ΔF0 of the two speakers) is recommended:
expect gain ≈ 0 for ΔF0 > 150 Hz, gain up to +2 dB for ΔF0 < 80 Hz.

### 4.2 Secondary metrics (16 kHz, CPU)

| Metric | Baseline | +F0 (proposed) | Note |
|---|---|---|---|
| PESQ (WB) | 2.5 – 2.9 | 2.6 – 3.1 | +0.1 – 0.3; tonal contour preservation should reduce warble |
| STOI | 0.90 – 0.94 | 0.91 – 0.95 | +0.005 – 0.02 |
| SDRi | ≈ SI-SNRi + 0.3 – 0.8 | same relation | SDR ≥ SI-SNR at same permutation |

### 4.3 Inference latency & memory (5 s @ 32 kHz, A100, batch 1, fp16)

| Metric | Baseline (MHSA) | Hybrid + F0 | Reasoning |
|---|---|---|---|
| Peak activation memory (intra+inter attention/scan) | ≈ 1.3 GB (attention matrices) | ≈ 0.3 GB (state buffers) | intra attention (B·C, 8, 250, 250) ≈ 318 M floats vs state (B·C, 512, 64) ≈ 21 M floats |
| Wall-clock latency | reference | **predicted 1.2 – 2.0× faster** | scan replaces the attention matmul + softmax; two directions add overhead, so the win is sub-linear |
| CPU (reference scan) | n/a | **not production** (≈158 ms per 1 s audio even for a 24-dim toy model on Apple M-series) | needs a fused CPU kernel port |

Note: because dual-path bounds attention to per-chunk matrices (S=250, C≈159),
the hybrid's latency advantage grows with **utterance length** (C grows, scan
stays linear) and with **batch size** (attention memory grows with B, state
does not scale with B·C linearly in the same way). At 4–5 s the advantage is
real but modest; at 30 s+ it becomes large.

### 4.4 Training dynamics

| Quantity | Prediction |
|---|---|
| Early epochs (0–5 k steps) | Hybrid loss slightly *worse* than baseline (Mamba warm-up, LR 0.5×) |
| Late epochs | Hybrid catches up and overtakes by ≤ +1 dB |
| F0 variant | Slowest start (gate must learn), steepest improvement after ~10–20 epochs; check `f0_gate` weight norm — if ‖gate‖ → 0, conditioning is inert and 4.1's +F0 gain will be ≈ 0 |
| Wall-clock per step | Hybrid ≈ 1.1 – 1.5× baseline (2 scans + more projection FLOPs), partially offset by smaller memory footprint → larger effective batch |

### 4.5 Risk that predictions are wrong

- **±0.5 dB noise floor**: with ~500 test mixtures, a 0.5 dB difference may
  not be significant — run paired tests and report per-utterance distributions,
  not just means.
- **Gate collapse** (D5 in `disadvantages.md`): if F0 does not help, ablation 2
  ≈ ablation 3 — do not "explain away" a null result; report it.
- **Kernel variance**: fp16 scan numerics on different GPUs (A100 vs 3090) can
  shift SI-SNRi by ±0.2 dB — pin versions (see `requirements.txt`) and rerun
  the same checkpoint on both.

---

## 5. How to reproduce these numbers

```bash
# exact parameter counts
python - <<'EOF'
from conformer_bimamba.models import HybridSepformer
for name, kw in {"conformer": dict(block_type="conformer", use_f0=False, f0_dim=0),
                 "hybrid-noF0": dict(block_type="hybrid", use_f0=False, f0_dim=0),
                 "hybrid-F0":   dict(block_type="hybrid", use_f0=True,  f0_dim=64)}.items():
    m = HybridSepformer(encoder_dim=256, d_ffn=1024, conv_kernel=31, d_state=64,
                        expand=2, num_intra=4, num_inter=4, **kw)
    print(name, f"{sum(p.numel() for p in m.parameters())/1e6:.3f} M")
EOF

# actual results (once trained)
python evaluate.py --config configs/train_bimamba_f0.yaml \
    --checkpoint runs/bimamba-f0/checkpoints/best.pt \
    --mixtures mixtures/test --perceptual --latency-repeats 20
```
