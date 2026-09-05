# Phân tích Tham số & Dự đoán Kết quả

**Hybrid Conformer-BiMamba so với Vanilla Conformer (baseline)**

Mọi số đếm dưới đây được **đo** bằng cách khởi tạo các mô hình trong repository
này với cấu hình mặc định (`configs/*.yaml`: `d_model=256`, `d_ffn=1024`,
`conv_kernel=31`, `d_state=64`, `expand=2`, `f0_dim=64`, 4 khối intra + 4 khối
inter). Số liệu là tham số huấn luyện được trừ khi có ghi chú.

---

## 1. Tổng tham số

| Mô hình | Tham số | So với baseline | So với Sepformer* |
|---|---|---|---|
| **Vanilla Conformer (MHSA)** — baseline | **10.740.224 (10,74 M)** | — | cỡ Sepformer-base |
| **Hybrid Bi-Mamba (không F0)** — ablation 2 | **16.294.400 (16,29 M)** | +5,55 M (+51,7 %) | cỡ Sepformer-large |
| **Hybrid Bi-Mamba + F0** — đề xuất | **16.430.816 (16,43 M)** | +5,69 M (+53,0 %) | cỡ Sepformer-large |

\* Bối cảnh tài liệu: Sepformer-base ≈ 2,6 M, Sepformer-large ≈ 26 M (Subakan
et al., 2021); TF-GridNet ≈ 14 M (Wang et al., 2023). Bản hybrid của chúng ta
nằm giữa Sepformer-base và Sepformer-large — cùng dải với TF-GridNet.

**Kết luận chính:** việc đổi MHSA→Bi-Mamba *làm tăng* tham số ~53 % ở cùng độ
rộng. Lợi thế O(n) nói về **tỉ lệ tính toán/bộ nhớ theo chiều dài chuỗi**, không
phải là mạng nhỏ hơn (xem `disadvantages_vi.md` D1).

### Chi phí của điều kiện hóa F0 (thanh điệu)
- Cổng F0 mỗi lớp: 8 lớp × 2 chiều × 8.320 = **133.120**
- `F0Projector` dùng chung: **3.296**
- **Toàn bộ bộ máy F0: +136.416 tham số = +0,84 %** so với hybrid không điều
  kiện hóa — gần như miễn phí.

---

## 2. Phân rã theo thành phần (mô hình đề xuất)

| Thành phần | Tham số | Tỉ trọng | Ghi chú |
|---|---|---|---|
| Encoder (Conv1d 1→256, k16/s8) | 4.096 | 0,02 % | chỉ trọng số (không bias) |
| Decoder (ConvTranspose1d) | 4.096 | 0,02 % | chỉ trọng số |
| Masker (Linear 256→512) | 131.584 | 0,8 % | |
| F0Projector (3→32→32→64) | 3.296 | 0,02 % | dùng chung, rẻ |
| **Chồng dual-path (8 khối)** | **16.287.744** | **99,1 %** | chiếm ưu thế |
| **Tổng** | **16.430.816** | 100 % | |

### Theo khối: HybridBlock so với ConformerBlock

| Lớp con | HybridBlock | ConformerBlock | Giống nhau? |
|---|---|---|---|
| FFN1 + FFN2 (gồm 2× LayerNorm) | 1.052.160 | 1.052.160 | ✅ giống hệt |
| DepthwiseConv1d k31 (gồm LayerNorm, BatchNorm) | 9.216 | 9.216 | ✅ giống hệt |
| **Lớp con Bi-Mamba** (lõi fwd+bwd, gồm LayerNorm) | **974.592** | — | thay thế MHSA |
| **Lớp con MHSA** (gồm LayerNorm) | — | **263.680** | đã bỏ |
| **Tổng khối** | **2.035.968** | **1.325.056** | Δ = +710.912/khối |

### Giải phẫu một lõi Bi-Mamba (mỗi chiều; ×2 cho fwd+bwd)

| Tham số | Kích thước | Số lượng |
|---|---|---|
| `in_proj` | 256 → 2·512 | 262.144 |
| `out_proj` | 512 → 256 | 131.072 |
| `x_proj` | 512 → dt_rank(16)+2·d_state(128) | 73.728 |
| `dt_proj` | 16 → 512 | 8.704 |
| `f0_gate` | 64 → 2·d_state(128) | 8.320 |
| `conv1d` (d_conv=4, groups=512) | 512×4 + 512 | 2.560 |
| `D` | 512 | 512 |
| **lõi huấn luyện được** | | **487.040** |
| `A_log` (buffer, không huấn luyện) | 512×64 | 32.768 |

Tương đương MHSA: `in_proj` 196.608 + `out_proj` 65.792 + bias 768 ≈ 263 k.

---

## 3. Công thức tham số (mọi cấu hình)

Ký hiệu `d` = encoder_dim, `f` = d_ffn, `k` = conv_kernel, `d_s` = d_state,
`e` = expand, `r` = dt_rank = ⌈d/16⌉, `f0` = f0_dim, `N` = số khối.

```
Encoder   = d · 16
Decoder   = d · 16
Masker    = d · 2d
F0Proj    = 3·32 + 32 + 32·32 + 32 + 32·64 + 64          (= 3.296 với f0=64)

Mỗi ConformerBlock:
  FFN×2    ≈ 2 · (2·d·f)
  MHSA     ≈ 4d² + 4d
  DWConv   ≈ k·d + 5d
Mỗi HybridBlock:
  FFN×2    ≈ 2 · (2·d·f)          (giống hệt)
  BiMamba  ≈ 2 · [ 2·(e·d)·d          (in_proj)
                 + (e·d)·(r + 2·d_s)  (x_proj)
                 + (r+1)·(e·d)        (dt_proj)
                 + (e·d)·d            (out_proj)
                 + (f0+1)·2·d_s       (f0_gate)
                 + 4·(e·d) + (e·d) ]  (conv1d + D)
  DWConv   ≈ k·d + 5d            (giống hệt)
```

---

## 4. Dự đoán về kết quả đầu ra

Đây là các **giả thuyết có thể bác bỏ** với dải số dựa trên tài liệu và phân
tích ở trên; phải được xác nhận bằng các lượt chạy `evaluate.py` trên tập kiểm
cố định (500 hỗn hợp, 0 dB, 2 người nói, 32 kHz).

### 4.1 Chỉ số chính — SI-SNRi (dB cải thiện so với tín hiệu trộn)

| Mô hình | SI-SNRi dự đoán | Lý do |
|---|---|---|
| Vanilla Conformer (baseline) | **9,0 – 12,0** | Sepformer-large đạt 20,4 dB trên WSJ0-2mix (8 kHz, 0 dB, 30 h huấn luyện, 26 M tham số). Bối cảnh của ta khó hơn mỗi câu (32 kHz, 4–5 s, Vivos ~15 h + Common Voice) và mô hình 10,7 M → dự kiến khoảng một nửa dư địa. |
| Hybrid Bi-Mamba (không F0) | **9,5 – 12,5** | Các bộ tách giọng dựa Mamba (ví dụ MOSSFormer, ~20,8 dB trên WSJ0-2mix ở kích thước tương tự) ngang Transformer; dự kiến **+0,0 – +0,8 dB** so với baseline, tập trung ở câu dài hơn nơi trạng thái xa của phép quét phát huy. |
| Hybrid Bi-Mamba + F0 (đề xuất) | **10,0 – 13,5** | Dự kiến **+0,5 – +1,5 dB so với hybrid không-F0**, *nếu* cổng F0 được kích hoạt. Lợi ích lớn nhất ở các cặp **cùng giới / cao độ chồng lấn** và ở **từ mang thanh điệu** (đường nét được giữ); gần như không lợi ở cặp khác giới, tách tốt. |

Giả thuyết tinh chỉnh: điều kiện hóa F0 giúp nhiều nhất nơi attention/Mamba
một mình nhầm người nói — tức cặp có dải F0 chồng lấn. Khuyến nghị **phân tích
có điều kiện** (chia cặp kiểm theo ΔF0 của hai người nói): dự kiến lợi ích ≈ 0
khi ΔF0 > 150 Hz, lợi ích tới +2 dB khi ΔF0 < 80 Hz.

### 4.2 Chỉ số phụ (16 kHz, CPU)

| Chỉ số | Baseline | +F0 (đề xuất) | Ghi chú |
|---|---|---|---|
| PESQ (WB) | 2,5 – 2,9 | 2,6 – 3,1 | +0,1 – 0,3; giữ đường nét thanh điệu nên giảm hiện tượng rè (warble) |
| STOI | 0,90 – 0,94 | 0,91 – 0,95 | +0,005 – 0,02 |
| SDRi | ≈ SI-SNRi + 0,3 – 0,8 | quan hệ tương tự | SDR ≥ SI-SNR ở cùng hoán vị |

### 4.3 Độ trễ suy luận & bộ nhớ (5 s @ 32 kHz, A100, batch 1, fp16)

| Chỉ số | Baseline (MHSA) | Hybrid + F0 | Lý do |
|---|---|---|---|
| Bộ nhớ activation đỉnh (ma trận attention intra+inter / state) | ≈ 1,3 GB (ma trận attention) | ≈ 0,3 GB (buffer trạng thái) | attention intra (B·C, 8, 250, 250) ≈ 318 M số thực so với state (B·C, 512, 64) ≈ 21 M số thực |
| Độ trễ wall-clock | tham chiếu | **dự đoán nhanh hơn 1,2 – 2,0×** | phép quét thay matmul attention + softmax; hai chiều thêm chi phí nên thắng lợi dưới tuyến tính |
| CPU (phép quét tham chiếu) | không áp dụng | **không dùng sản xuất được** (≈158 ms mỗi 1 s âm thanh ngay cả với mô hình đồ chơi 24-dim trên Apple M-series) | cần port kernel hợp nhất cho CPU |

Lưu ý: vì dual-path chặn attention ở ma trận theo chunk (S=250, C≈159), lợi thế
độ trễ của hybrid tăng theo **độ dài câu** (C tăng, phép quét vẫn tuyến tính) và
theo **batch size** (bộ nhớ attention tăng theo B, trạng thái không tăng tuyến
tính cùng B·C theo cách tương tự). Ở 4–5 s lợi thế có thật nhưng khiêm tốn; ở
30 s+ nó trở nên lớn.

### 4.4 Diễn biến huấn luyện

| Đại lượng | Dự đoán |
|---|---|
| Epoch đầu (0–5 k bước) | Loss hybrid hơi *tệ hơn* baseline (Mamba cần warm-up, LR 0,5×) |
| Epoch muộn | Hybrid bắt kịp và vượt ≤ +1 dB |
| Biến thể F0 | Khởi đầu chậm nhất (cổng phải học), cải thiện dốc nhất sau ~10–20 epoch; kiểm tra chuẩn `f0_gate` — nếu ‖gate‖ → 0, điều kiện hóa vô hiệu và lợi ích +F0 ở 4.1 sẽ ≈ 0 |
| Wall-clock mỗi bước | Hybrid ≈ 1,1 – 1,5× baseline (2 phép quét + nhiều FLOP chiếu hơn), bù một phần nhờ footprint bộ nhớ nhỏ hơn → batch hiệu dụng lớn hơn |

### 4.5 Rủi ro dự đoán sai

- **Sàn nhiễu ±0,5 dB**: với ~500 hỗn hợp kiểm, chênh lệch 0,5 dB có thể không
  có ý nghĩa — chạy kiểm định ghép cặp và báo phân phối theo từng câu, không
  chỉ trung bình.
- **Sụp đổ cổng** (D5 trong `disadvantages_vi.md`): nếu F0 không giúp ích,
  ablation 2 ≈ ablation 3 — đừng "giải thích cho qua" kết quả vô hiệu; hãy báo
  cáo nó.
- **Phương sai kernel**: số học quét fp16 trên GPU khác nhau (A100 vs 3090) có
  thể dịch SI-SNRi ±0,2 dB — khóa phiên bản (xem `requirements.txt`) và chạy
  lại cùng checkpoint trên cả hai.

---

## 5. Cách tái tạo các con số này

```bash
# số tham số chính xác
python - <<'EOF'
from conformer_bimamba.models import HybridSepformer
for name, kw in {"conformer": dict(block_type="conformer", use_f0=False, f0_dim=0),
                 "hybrid-noF0": dict(block_type="hybrid", use_f0=False, f0_dim=0),
                 "hybrid-F0":   dict(block_type="hybrid", use_f0=True,  f0_dim=64)}.items():
    m = HybridSepformer(encoder_dim=256, d_ffn=1024, conv_kernel=31, d_state=64,
                        expand=2, num_intra=4, num_inter=4, **kw)
    print(name, f"{sum(p.numel() for p in m.parameters())/1e6:.3f} M")
EOF

# kết quả thực tế (sau khi huấn luyện)
python evaluate.py --config configs/train_bimamba_f0.yaml \
    --checkpoint runs/bimamba-f0/checkpoints/best.pt \
    --mixtures mixtures/test --perceptual --latency-repeats 20
```
