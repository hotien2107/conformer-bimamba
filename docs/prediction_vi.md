# Dự đoán: Thời gian huấn luyện, Kết quả theo tham số & So sánh với các mô hình liên quan

*Tài liệu dự đoán định lượng cho mô hình Hybrid Conformer-BiMamba. Số tham số và
MACs là **số đo thực tế** từ code trong repository; thời gian huấn luyện và
SI-SNRi là **ước tính** dựa trên công thức FLOPs + dữ liệu công bố của các mô
hình cùng dòng (DPMamba, SPMamba, SepMamba, MOSSFormer2, TF-GridNet, SepFormer).
Tất cả ước tính phải được xác nhận bằng các lần chạy thật.*

---

## 1. Giả định chung (cố định cho mọi ước tính)

| Giả định | Giá trị | Ghi chú |
|---|---|---|
| Phần cứng | 1× RTX 3090 / A10G (24 GB); so sánh thêm A100 (40 GB) | theo ràng buộc dự án |
| Dữ liệu huấn luyện | Vivos train (11.660) + Common Voice vi (≈ 40.000) ≈ **50.000 câu** | dynamic mixing, 4 s @ 32 kHz |
| Batch hiệu dụng | **8** (batch 4 × grad accum 2) | riêng biến thể L dùng 4 (batch 2 × 2) |
| Bước/epoch | 50.000 / 8 = **6.250** (L: 12.500) | |
| AMP | fp16 + GradScaler | fp32 cho `dt` |
| Hiệu suất GPU thực dụng | 3090: ~15 TFLOPS; A100: ~30 TFLOPS | FFN chiếm ưu thế (matmul); phép quét Mamba bị giới hạn băng thông |
| Epochs hội tụ (dự đoán) | 70–130 tùy kích thước, early stop patience 10 | Vivos nhỏ → quá khớp là ràng buộc chính |

---

## 2. Các biến thể mô hình (số đo thực tế)

| Biến thể | encoder_dim | d_ffn | Khối (intra+inter) | d_state/expand/f0 | **Tham số** | **MACs (G/s)** |
|---|---|---|---|---|---|---|
| Hybrid-F0-XS | 128 | 512 | 2+2 | 32 / 1 / 32 | **1,61 M** | 6,6 |
| Hybrid-F0-S | 192 | 768 | 3+3 | 48 / 2 / 48 | **6,98 M** | 29,3 |
| **Hybrid-F0-M (mặc định)** | 256 | 1024 | 4+4 | 64 / 2 / 64 | **16,43 M** | 69,1 |
| Hybrid-F0-L | 384 | 1536 | 6+6 | 96 / 2 / 96 | **55,10 M** | 232,1 |
| Conformer-M (baseline) | 256 | 1024 | 4+4 | MHSA, 8 head | **10,74 M** | 35,4 |
| Hybrid-noF0-M (ablation 2) | 256 | 1024 | 4+4 | 64 / 2 / 0 | **16,29 M** | 69,1 |

Lưu ý: MACs của hybrid gần gấp đôi Conformer cùng cấu hình dù chỉ hơn ~53 %
tham số — do `in_proj`/`out_proj` của hai lõi BiMamba (FFN vẫn chiếm phần lớn
ở cả hai). Đây là chi phí phải trả cho việc thay MHSA; bù lại bộ nhớ attention
O(n²) được thay bằng trạng thái O(n).

---

## 3. Ước tính thời gian huấn luyện

**Công thức:** `thời gian/step ≈ MACs_mỗi_step / TFLOPS_hiệu_dụng + 0,05 s chi phí`
với `MACs_mỗi_step = G/s × 4 s × batch`.

| Biến thể | MACs/step (G) | Step (3090) | Step (A100) | Epoch (3090) | Epoch (A100) | Epochs dự đoán | **Tổng 3090** | **Tổng A100** |
|---|---|---|---|---|---|---|---|---|
| Hybrid-F0-XS | 106 | ~0,06 s | ~0,04 s | ~6 min | ~4 min | 70–90 | **7–9 h** | 4–6 h |
| Hybrid-F0-S | 469 | ~0,10 s | ~0,06 s | ~10 min | ~6 min | 80–100 | **13–17 h** | 8–10 h |
| **Hybrid-F0-M** | 1.106 | **0,2–0,4 s** | 0,1–0,2 s | **21–42 min** | 10–21 min | 90–120 | **1,5–3,5 ngày** | 0,8–1,8 ngày |
| Hybrid-F0-L* | 1.856 | 0,3–0,5 s | 0,2–0,3 s | 63–104 min | 42–63 min | 100–130 | **4,4–9,4 ngày** | 2,9–5,7 ngày |
| Conformer-M | 566 | 0,13–0,2 s | 0,08–0,12 s | 14–21 min | 8–12 min | 80–110 | **19–39 h** | 11–22 h |
| Hybrid-noF0-M | 1.106 | 0,2–0,4 s | 0,1–0,2 s | 21–42 min | 10–21 min | 90–120 | 1,5–3,5 ngày | 0,8–1,8 ngày |

\* L dùng batch hiệu dụng 4 (batch 2 × 2) vì VRAM; kèm chi phí mỗi lần đánh giá
(val 300 hỗn hợp) ~2–5 min mỗi 2 epoch.

**Kết luận thời gian:** với ràng buộc 24 GB / 3–5 ngày của dự án, **Hybrid-F0-M
(16,43 M) là điểm ngọt**: ~1,5–3,5 ngày trên 3090 (đúng mục tiêu), ~1 ngày trên
A100. Biến thể L (~55 M) **không khả thi** trên 24 GB (4–9 ngày + VRAM sát
ngưỡng); XS/S phù hợp cho khảo sát nhanh (dưới 1 ngày).

---

## 4. Ước tính VRAM (batch 4, câu 4 s @ 32 kHz, fp16, không checkpointing)

| Biến thể | VRAM dự đoán | Khả năng trên 24 GB |
|---|---|---|
| Hybrid-F0-XS | ~2 GB | thoải mái (batch tới 16) |
| Hybrid-F0-S | ~4 GB | thoải mái (batch tới 8) |
| Hybrid-F0-M | ~6–9 GB | thoải mái (batch 4–6; có thể batch 8) |
| Hybrid-F0-L | ~20 GB (batch 2) | sát ngưỡng → cần checkpointing |
| Conformer-M | ~8–10 GB | thoải mái (ma trận attention intra ≈ 0,65 GB fp16 tại batch 4) |

Điểm mạnh của phép quét: thay ma trận attention (B·C, 8, 250, 250) ≈ 318 M số
thực bằng buffer trạng thái (B·C, 512, 64) ≈ 21 M — nên hybrid-M dùng bộ nhớ
*thấp hơn* Conformer-M dù nhiều tham số hơn, và có thể tăng batch.

---

## 5. Dự đoán kết quả theo từng mức tham số

### 5.1 Trên Vivos tiếng Việt (32 kHz, 2 người nói, SNR U[0,5] dB, 4–5 s)

| Biến thể | Tham số | SI-SNRi dự đoán | Lý do |
|---|---|---|---|
| Hybrid-F0-XS | 1,61 M | 6,0 – 8,0 | đủ sức cho task đơn giản, thiếu dung lượng cho giọng chồng lấn |
| Hybrid-F0-S | 6,98 M | 8,0 – 10,5 | tương đương DPRNN-class trên dữ liệu nhỏ |
| **Hybrid-F0-M** | **16,43 M** | **10,0 – 13,5** | kỳ vọng chính; +0,5–1,5 dB nhờ F0 trên cặp cùng giới |
| Hybrid-F0-L | 55,10 M | 11,5 – 15,0 | cao nhất nhưng rủi ro quá khớp Vivos (~15 h) cao; lợi ích co lại |
| Conformer-M (baseline) | 10,74 M | 9,0 – 12,0 | điểm chuẩn để đo mức tăng |
| Hybrid-noF0-M | 16,29 M | 9,5 – 12,5 | tách riêng giá trị của F0 (chênh với M = +0,5–1,5) |

### 5.2 Quy đổi tương đương WSJ0-2Mix (8 kHz, 200 epoch, dynamic mixing — giả định)

Để định vị với bảng so sánh quốc tế (không phải kết quả thật trên tiếng Việt),
dùng đường cong chuẩn từ DPMamba: (2,3 M → 19,2 dB), (8,1 M → 21,4 dB),
(15,9 M → 22,6 dB):

| Biến thể | Tham số | SI-SNRi WSJ0 tương đương (dự đoán) | Đối chiếu |
|---|---|---|---|
| Hybrid-F0-XS | 1,61 M | 18,5 – 19,5 | ≈ DPMamba-XS (19,2) |
| Hybrid-F0-S | 6,98 M | 20,8 – 21,6 | ≈ DPMamba-S (21,4) |
| Hybrid-F0-M | 16,43 M | 22,3 – 22,8 | ≈ DPMamba-M (22,6); kém SepMamba-M 22,7 cùng cỡ |
| Hybrid-F0-L | 55,10 M | 23,5 – 24,2 | ≈ MOSSFormer2-L (24,1) |
| Conformer-M | 10,74 M | 20,0 – 21,0 | giữa SepFormer (20,4/26 M) và MossFormer-S (20,9/10,8 M) |

---

## 6. So sánh trực tiếp với các mô hình liên quan (cùng benchmark 8 kHz)

| Mô hình | Tham số | MACs (G/s) | SI-SNRi | Bộ nhớ | Thời gian huấn luyện tham khảo |
|---|---|---|---|---|---|
| DPMamba-XS | 2,3 M | — | 19,2 | — | 200 epoch, WSJ0-2mix, L40 |
| DPMamba-S | 8,1 M | — | 21,4 | — | như trên |
| DPMamba-M | 15,9 M | — | 22,6 | — | như trên |
| SPMamba | 6,14 M | 238,69 | 22,5 | 14,4 GB | 200 epoch, WSJ0-2mix |
| SepMamba-S (+DM) | 7,2 M | 12,46 | 21,2 | 2,0 GB | 200 epoch |
| SepMamba-M (+DM) | 22 M | 37,0 | 22,7 | 3,04 GB | 200 epoch |
| MOSSFormer2 (+DM) | 55,7 M | 84,2 | **24,1** | 12,3 GB | 200 epoch |
| TF-GridNet (L) | 14,4 M | 231,1 | 23,4 | — | 200 epoch |
| SepFormer (+DM) | 26 M | 257,94 | 20,4 / 22,3 | 35,3 GB | 200 epoch |
| S4M | 3,6 M | — | 20,5 | — | — |
| **Hybrid-F0-M (đề xuất)** | **16,43 M** | **69,1** | *chưa đo* | **~6–9 GB** | **90–120 epoch ≈ 1,5–3,5 ngày (3090)** |

Nhận xét rút ra từ bảng:
- **Chi phí**: MACs của ta (69,1 G/s) thấp hơn hẳn SepFormer (258) và TF-GridNet
  (231), ngang MOSSFormer2 (84) dù ít tham số hơn 3,4×; bộ nhớ ~6–9 GB thấp hơn
  SepFormer (35 GB) rất nhiều.
- **Hiệu năng kỳ vọng**: đứng giữa DPMamba-M và SepMamba-M — hợp lý vì cùng cỡ
  tham số; không kỳ vọng vượt TF-GridNet/MOSSFormer2 (miền T-F + tham số lớn).
- **Điểm khác biệt duy nhất**: F0 conditioning — không mô hình nào trong bảng
  có; lợi ích dự đoán +0,5–1,5 dB chỉ xuất hiện trên tiếng Việt/ngôn ngữ thanh
  điệu, không đo được trên WSJ0-2mix (tiếng Anh, ít thanh điệu).

---

## 7. Kịch bản khuyến nghị (24 GB, 3–5 ngày)

1. **Chạy trước (ngày 1–2, XS/S)**: khảo sát nhanh + kiểm chứng pipeline —
   Hybrid-F0-S (~15 h) đủ để xác nhận F0 gate hoạt động (‖gate‖ > 0) và so với
   Conformer-S.
2. **Chạy chính (ngày 2–5, M)**: 3 cấu hình `train_bimamba_f0.yaml`,
   `train_bimamba.yaml`, `train_conformer.yaml` song song nếu có đủ VRAM/luồng;
   nếu không, chạy tuần tự baseline → noF0 → F0 (ưu tiên F0 cuối vì cần so sánh).
3. **Báo cáo**: SI-SNRi/SDRi + PESQ/STOI + latency (GPU/CPU) trên 500 hỗn hợp
   test cố định; phân tích có điều kiện theo ΔF0 của cặp người nói.

---

## 8. Rủi ro dự đoán sai & cách kiểm chứng

| Rủi ro | Ảnh hưởng | Kiểm chứng |
|---|---|---|
| Hiệu suất GPU thực dụng thấp hơn 15 TFLOPS (phép quét băng thông) | Thời gian M tăng lên 4–5 ngày | đo thời gian/step thật ở 50 bước đầu; điều chỉnh `max_steps` |
| Vivos quá nhỏ → quá khớp sớm, lợi ích F0/L biến mất | SI-SNRi thấp hơn dải dự đoán | theo dõi gap train/val; early stop; thêm Common Voice |
| F0 gate sụp đổ (‖gate‖→0) | M ≈ noF0 | log chuẩn `f0_gate.weight` mỗi epoch; so ablation 2 vs 3 |
| Dynamic mixing chưa đủ (50 k câu) | kém hơn các mô hình 200-epoch WSJ0 | không so trực tiếp; chỉ so trên cùng tập test Vivos |
| fp16 gây underflow `dt` | loss NaN giữa chừng | `amp=true` nhưng giữ `dt_proj` fp32; fallback fp32 nếu NaN |

---

## 9. Nguồn số liệu đối chiếu

- [DPMamba (arXiv 2403.18257)](https://arxiv.org/abs/2403.18257) — đường cong tham số → SI-SNRi (XS/S/M).
- [SPMamba (arXiv 2404.02063)](https://arxiv.org/abs/2404.02063) + [GitHub](https://github.com/JusperLee/SPMamba) — 22,5 dB / 6,14 M / 238,69 G/s.
- [SepMamba (arXiv 2410.20997)](https://arxiv.org/abs/2410.20997) + [GitHub](https://github.com/andrasschin/SepMamba) — bảng MACs/latency/bộ nhớ của SepFormer, MossFormer S/M/L, MossFormer2, TF-GridNet S/M/L.
- [MOSSFormer2 (arXiv 2312.11825)](https://arxiv.org/abs/2312.11825); [TF-GridNet (arXiv 2211.12433)](https://arxiv.org/abs/2211.12433); SepFormer (ICASSP 2021).
- Tham số/MACs của các biến thể trong mục 2: đo trực tiếp từ repository này (xem `parameter_analysis_vi.md`).
