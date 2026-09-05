# So sánh chi tiết: Hybrid Conformer-BiMamba với các mô hình tương tự

*Tài liệu so sánh mô hình đề xuất (Hybrid Conformer-BiMamba + điều kiện hóa F0)
với các mô hình tách giọng nói hiện đại, đặc biệt là nhóm dựa trên
Mamba/State-Space Model (SSM): MOSSFormer, MOSSFormer2, SPMamba, SepMamba,
Dual-path Mamba (DPMamba), Mamba-TasNet, S4M…*

> **Ghi chú về số liệu:** mọi con số SI-SNRi/SDRi/tham số/MACs dưới đây được
> lấy từ bài báo gốc hoặc bảng so sánh công bố chính thức của các tác giả
> (có link nguồn ở mục 8). Các benchmark chuẩn (WSJ0-2Mix, WHAM!, Libri2Mix)
> đều ở **8 kHz**, trong khi mô hình đề xuất nhắm tới **32 kHz tiếng Việt** —
> con số không so sánh trực tiếp được, chỉ dùng để định vị tương đối.

---

## 1. Bảng so sánh tổng quan (benchmark WSJ0-2Mix, 2 người nói, 8 kHz)

| Mô hình | Năm | Cơ chế phụ thuộc xa | SI-SNRi (dB) | SDRi (dB) | Tham số | MACs (G/s) | Bộ nhớ (GB) |
|---|---|---|---|---|---|---|---|
| Conv-TasNet | 2019 | Conv + LSTM cục bộ | 15,3 | 15,6 | 5,1 M | 2,82 | 1,13 |
| DualPathRNN (DPRNN) | 2020 | RNN hai chiều (dual-path) | 18,8 | 19,0 | 2,6 M | 42,52 | 7,31 |
| SepFormer (+DM) | 2021 | Transformer dual-path | 20,4 / 22,3 | — | 26 M | 257,94 | 35,30 |
| S4M (SSM thuần) | 2024 | Selective SSM (Mamba-1) | 20,5 | 20,7 | 3,6 M | — | — |
| MOSSFormer (S) | 2024 | Gated single-head transformer | — | 20,9 | 10,8 M | — | — |
| MOSSFormer (L) + DM | 2024 | Bottleneck-composite attention | — | 22,8 | 42,1 M | 70,4 | 9,57 |
| MOSSFormer2 + DM | 2024 | Transformer + GFRN (Mamba-inspired) | — | **24,1** | 55,7 M | 84,2 | 12,30 |
| TF-GridNet (L) | 2023 | Full-band + sub-band (BLSTM) | 23,4 | 23,5 | 14,4 M | 231,1 | — |
| TF-GridNet (M) | 2023 | như trên | 22,2 | — | 8,4 M | 36,2 | — |
| **SPMamba** | 2024 | TF-GridNet, **BLSTM → BiMamba** | **22,5** | 22,7 | **6,14 M** | 238,69 | 14,40 |
| **SepMamba (S) + DM** | 2025 | U-Net + **BiMamba** | 21,2 | 21,4 | 7,2 M | **12,46** | **2,00** |
| **SepMamba (M) + DM** | 2025 | U-Net + BiMamba | 22,7 | 22,9 | 22 M | 37,0 | 3,04 |
| **DPMamba (XS)** | 2024 | **Dual-path BiMamba** | 19,2 | 19,4 | 2,3 M | — | — |
| **DPMamba (S)** | 2024 | Dual-path BiMamba | 21,4 | 21,6 | 8,1 M | — | — |
| **DPMamba (M)** | 2024 | Dual-path BiMamba | 22,6 | 22,7 | 15,9 M | — | — |
| QDPN | 2023 | Quasi-dual-path, transformer | 23,6 | — | 200 M | — | — |
| **Hybrid Conformer-BiMamba (đề xuất)** | 2025 | **Dual-path BiMamba + FFN + DWConv + F0** | *chưa huấn luyện* | — | **16,43 M** | — | — |

Nguồn: [SepMamba README](https://github.com/andrasschin/SepMamba) (bảng so sánh
chính thức), [DPMamba (arXiv 2403.18257)](https://arxiv.org/abs/2403.18257),
[SPMamba (arXiv 2404.02063)](https://arxiv.org/abs/2404.02063).

**Đọc nhanh:** cùng cỡ tham số ~16 M với đề xuất của chúng ta, DPMamba-M đạt
**22,6 dB**; SepMamba-M (22 M) đạt 22,7 dB; SPMamba chỉ 6,14 M đạt 22,5 dB —
tức các mô hình Mamba song phương đã chứng minh hiệu năng **ngang hoặc hơn
transformer** với chi phí thấp hơn rõ rệt về MACs và bộ nhớ.

---

## 2. Chi tiết từng mô hình tương tự

### 2.1 DPMamba — Dual-path Mamba (gần nhất với đề xuất)
- **Bài báo:** [Dual-path Mamba: Short and Long-term Bidirectional Selective Structured State Space Models for Speech Separation](https://arxiv.org/abs/2403.18257) (Jiang et al., 2024), cùng nhóm với Mamba-TasNet.
- **Ý tưởng:** giống hệt khung dual-path của Sepformer/DPRNN nhưng thay cả
  attention *lẫn* RNN bằng **đơn vị BiMamba** (hai chiều: thuận + nghịch) cho
  cả xử lý trong đoạn (intra-chunk, ngắn hạn) và liên đoạn (inter-chunk, dài
  hạn); chuẩn hóa RMSNorm + skip connection.
- **Kết quả:** XS 19,2 dB/2,3 M; S 21,4/8,1 M; M 22,6/15,9 M — **vượt Sepformer
  và DPRNN cùng cỡ tham số**, dù tham số ít hơn.
- **Khác với đề xuất của chúng ta:**
  - Không có **FFN và Depthwise-Conv** (khối Conformer) — đề xuất của ta giữ
    bias cục bộ cho âm vị/phụ âm.
  - **Không có điều kiện hóa F0/thanh điệu** — điểm khác biệt cốt lõi của ta.
  - DPMamba dùng Mamba hai chiều **thuần**; ta dùng lõi Mamba-1 có B, C được
    điều biến bởi F0 (không thể làm điều này với kernel hợp nhất Mamba-2).
  - DPMamba huấn luyện 200 epoch trên WSJ0-2mix 8 kHz; ta nhắm Vivos 32 kHz.

### 2.2 SPMamba — State-space model is all you need
- **Bài báo:** [SPMamba (arXiv 2404.02063)](https://arxiv.org/abs/2404.02063) (Li, Chen, Hu — ĐH Thanh Hoa, 2024).
- **Ý tưởng:** giữ nguyên kiến trúc **TF-GridNet** (full-band + sub-band,
  mặt nạ phức) nhưng **thay toàn bộ BLSTM bằng mô-đun BiMamba** — mô hình hóa
  quan hệ không-thời gian theo cả hai chiều thời gian và tần số với độ phức
  tạp tuyến tính.
- **Kết quả:** WSJ0-2Mix **22,5 dB / 6,14 M / 238,69 G/s**; WHAM! 17,4; Libri2Mix
  19,9; Echo2Mix 15,3 (thắng TF-GridNet trên các tập khó hơn).
- **Khác với đề xuất:** miền **tần số-thời gian (STFT)** + mask phức, không
  phải miền thời gian dual-path; không có F0; MACs cao (238 G/s — chủ yếu do
  STFT full-band/sub-band) dù tham số ít.

### 2.3 SepMamba — U-Net với BiMamba
- **Bài báo:** [SepMamba (arXiv 2410.20997)](https://arxiv.org/abs/2410.20997) (DTU, ICASSP 2025).
- **Ý tưởng:** kiến trúc **U-Net** trong miền thời gian, các lớp **BiMamba** ở
  từng độ phân giải; có biến thể nhân quả cho ứng dụng real-time.
- **Kết quả:** S 21,2 dB/7,2 M/**12,46 G/s**/**2,00 GB**; M 22,7/22 M/37 G/s/
  3,04 GB — **rẻ hơn hẳn Sepformer (258 G/s, 35 GB) với kết quả cao hơn**.
- **Khác với đề xuất:** U-Net thay vì dual-path; không FFN/DWConv kiểu
  Conformer; không F0; nhưng cho thấy BiMamba + gating (SiLU) là thành phần
  đủ mạnh để thay attention.

### 2.4 Mamba-TasNet / Speech Slytherin
- **Bài báo:** [Mamba-TasNet (arXiv 2407.09732)](https://arxiv.org/abs/2407.09732) — cùng nhóm DPMamba; khảo sát Mamba cho tách giọng/nhận dạng/tổng hợp.
- **Ý tưởng:** thay mạng LSTM/transformer trong khung TasNet bằng Mamba
  hai chiều; cảnh báo về **bất ổn số khi huấn luyện mô hình cỡ L** (khuyến
  nghị fp32, xem mục 6 rủi ro của ta).

### 2.5 S4M — Selective State Space Model thuần
- Từ bảng DPMamba: S4M-tiny 19,4 dB/1,8 M; S4M 20,5 dB/3,6 M — chứng minh
  SSM chọn lọc (Mamba-1) tự thân đã đạt ngang DPRNN với rất ít tham số.

### 2.6 MOSSFormer — Gated Single-Head Transformer
- **Bài báo:** [MOSSFormer (ICASSP 2024)](https://github.com/alibabasglab/MossFormer).
- **Ý tưởng:** attention **một head** + **bottleneck-composite self-attention**
  (mở rộng receptive field theo nhiều tần số) + cổng SiLU; giảm chi phí bậc
  hai của multi-head.
- **Kết quả:** S 20,9 dB/10,8 M; M 22,5/25,3 M; L 22,8/42,1 M (có dynamic
  mixing). Lưu ý: MOSSFormer **không dùng Mamba** — là "anh em" hiệu quả của
  attention, không phải SSM.

### 2.7 MOSSFormer2 — Transformer + GFRN (Mamba-inspired, không RNN)
- **Bài báo:** [MOSSFormer2 (arXiv 2312.11825)](https://arxiv.org/abs/2312.11825) (ICASSP 2024).
- **Ý tưởng:** kết hợp transformer (nhánh attention) với **GFRN — Gated Fully
  Recurrent Network**: mạng hồi quy *không có vòng lặp RNN*, dựa trên phép
  chiếu tuyến tính + tích chập, lấy cảm hứng từ cơ chế chọn lọc của Mamba để
  mô hình mẫu tuần tự mịn (fine-grained) song song hóa toàn chuỗi.
- **Kết quả:** **24,1 dB / 55,7 M / 84,2 G/s** — SOTA trên WSJ0-2Mix/3mix và
  Libri2Mix tại thời điểm công bố; nhưng **tham số gấp ~3,4 lần** đề xuất của
  ta và không có F0.

### 2.8 TF-GridNet — điểm tham chiếu SOTA (full-band + sub-band)
- **Bài báo:** [TF-GridNet (arXiv 2211.12433)](https://arxiv.org/abs/2211.12433).
- Ý tưởng: chia băng tần + BLSTM + attention hai chiều; L đạt 23,4 dB/14,4 M
  nhưng **231 G/s MACs, 445 G/s (bản espnet)** — đắt nhất trong nhóm; SPMamba
  ra đời chính để thay BLSTM của nó.

### 2.9 Sepformer — dual-path transformer (baseline chính của ta)
- Sepformer 20,4 dB (26 M, không dynamic mixing) / 22,3 (có DM); **257,94 G/s,
  35,3 GB bộ nhớ** — chi phí attention bậc hai rất cao; đây là mô hình mà đề
  xuất của ta thay MHSA bằng Bi-Mamba trong cùng khung dual-path.

### 2.10 SparseMamba & MambaFilter (cùng hướng, số liệu chưa kiểm chứng tại đây)
- **SparseMamba** (2024): lai **Mamba + sparse transformer** cho tách giọng
  đơn kênh — dùng attention thưa ở các vị trí quan trọng, Mamba cho phần còn
  lại; hướng đi gần với đề xuất (giảm bậc hai bằng SSM) nhưng vẫn giữ một phần
  attention và **không có F0**.
- **MambaFilter** (2024): dùng **channel attention kiểu Mamba** (quét chọn lọc
  trên kênh đặc trưng) cho tách giọng; rất nhẹ nhưng khác hẳn về vai trò so với
  BiMamba thay MHSA của ta.
- ⚠️ Hai mô hình này chưa được tôi kiểm chứng số liệu từ bản chính thức nên
  không đưa số cụ thể vào bảng 1.

---

## 3. So sánh kiến trúc chi tiết

| Tiêu chí | DPMamba | SPMamba | SepMamba | MOSSFormer2 | TF-GridNet | Sepformer | **Đề xuất (ta)** |
|---|---|---|---|---|---|---|---|
| Miền xử lý | Thời gian | STFT (T-F) | Thời gian | Thời gian | STFT (T-F) | Thời gian | Thời gian |
| Khung tổng thể | Dual-path | TF-GridNet | U-Net | Encoder-decoder | Full/sub-band | Dual-path | **Dual-path** |
| Thay thế attention bằng | BiMamba | BiMamba (thay BLSTM) | BiMamba | GFRN (Mamba-inspired) | BLSTM+attn | — | **BiMamba** |
| Giữ attention không? | Không | Không | Không | Có (song song) | Có | Có (MHSA) | **Không (bỏ hẳn MHSA)** |
| Bias cục bộ (Conv) | Không | Conv (full/sub-band) | Conv (U-Net) | Conv | Conv | DWConv | **DWConv kiểu Conformer** |
| FFN kiểu Conformer | Không | Không | Không | Có | Không | Có | **Có (giữ nguyên)** |
| Song phương | Có | Có | Có | — | Có | Có (attention) | **Có (2 lõi)** |
| Độ phức tạp | O(n) | O(n) (thay BLSTM) | O(n) | O(n) (GFRN) | cao (BLSTM) | O(n²) | **O(n)** |
| **Điều kiện hóa F0/thanh điệu** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (điểm mới duy nhất)** |
| Nhắm tiếng Việt | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (Vivos 32 kHz)** |
| Streaming | ✗ | ✗ | ✓ (biến thể) | ✗ | ✗ | ✗ | ✗ (offline) |

---

## 4. Vị trí của mô hình đề xuất — điểm giống & khác

### 4.1 Giống DPMamba nhất (cùng "gia đình" dual-path BiMamba)
Cả hai đều: chunk → intra (BiMamba) → inter (BiMamba) → overlap-add; hai chiều
thuận/nghịch; phi nhân quả; dùng kernel Mamba qua `mamba-ssm`.

### 4.2 Khác biệt so với DPMamba (3 điểm, theo thứ tự quan trọng)
1. **Điều kiện hóa F0 (thanh điệu)**: điều biến trực tiếp ma trận B, C của SSM
   theo đường nét F0 của tín hiệu trộn. **Không mô hình nào trong bảng có
   tính năng này** — đây là đóng góp mới duy nhất của đề xuất, hợp lý về mặt
   ngữ âm học cho tiếng Việt (6 thanh thể hiện qua F0 contour).
2. **Khối Conformer trọn vẹn**: FFN (expansion 4×) + DepthwiseConv giữ nguyên
   xung quanh BiMamba → giữ bias cục bộ (âm vị, phụ âm bật) mà DPMamba không
   có; ablation sạch với baseline Conformer thuần MHSA.
3. **Điều kiện kỹ thuật khác**: lõi Mamba-1 có B, C tường minh (bắt buộc để
   chèn F0; DPMamba cũng dùng Mamba-1 nhưng không điều biến B, C); mục tiêu
   32 kHz tiếng Việt thay vì 8 kHz WSJ.

### 4.3 Điểm yếu tương đối so với nhóm Mamba khác
- **Tham số cao hơn** SPMamba (16,4 M vs 6,14 M) và DPMamba-S (8,1 M) — do
  giữ FFN + DWConv và dùng `expand=2`.
- Không có biến thể nhân quả như SepMamba → chưa phủ streaming.
- Miền thời gian dual-path kém tự nhiên hơn SPMamba/TF-GridNet trong việc mô
  hình hóa cấu trúc tần số (không có sub-band).

---

## 5. Dự đoán hiệu năng tương đối (có cơ sở từ các mô hình cùng cỡ)

| Ngữ cảnh | Mô hình cùng cỡ tham số | Kết quả họ đạt | Kỳ vọng cho đề xuất |
|---|---|---|---|
| WSJ0-2Mix 8 kHz, huấn luyện đủ (200 epoch, DM) | DPMamba-M 15,9 M → 22,6 dB; SepMamba-M 22 M → 22,7 dB | 22,6–22,7 dB | **~21–23 dB** (16,4 M nằm giữa hai mức; thêm FFN/DWConv + F0 có thể bù cho tham số ít hơn SepMamba-M) |
| Cùng tập (giả định) — lợi ích F0 | — | — | **+0,5–1,5 dB** trên các cặp cùng giới/ΔF0 thấp so với phiên bản không F0 |
| Vivos tiếng Việt 32 kHz (dữ liệu nhỏ ~15 h) | — | — | **10–13,5 dB** (xem `parameter_analysis_vi.md` §4) — dữ liệu ít hơn WSJ0 rất nhiều nên con số thấp hơn hẳn |

Nhận định quan trọng: kết quả trên WSJ0-2mix của các mô hình Mamba đã **xác
nhận giả thuyết cốt lõi** của đề xuất — BiMamba dual-path đủ mạnh để thay
attention với chi phí thấp hơn. Phần còn lại cần kiểm chứng riêng là **giá trị
của điều kiện hóa F0 trên tiếng Việt**, điều chưa mô hình nào làm.

---

## 6. Rủi ro tham khảo từ các mô hình tiền nhiệm

1. **Bất ổn số khi huấn luyện**: Mamba-TasNet/DPMamba ghi nhận mô hình cỡ L
   mất ổn định, khuyến nghị fp32 + kỹ thuật ổn định loss (Jamba §6.4) → ta đã
   áp dụng `mamba_lr_scale=0.5`, warmup dài, grad clip.
2. **Dynamic mixing là yếu tố quyết định**: SepFormer +DM 22,3 vs 20,4 không
   DM; MossFormer (L) 22,8 có DM → pipeline của ta đã dùng dynamic mixing.
3. **Miền tần số vẫn mạnh**: TF-GridNet/SPMamba (T-F) đang dẫn đầu bảng; nếu
   kết quả dual-path thời gian thấp hơn kỳ vọng, hướng mở rộng là lai
   full-band/sub-band + BiMamba + F0.

---

## 7. Kết luận

Mô hình đề xuất nằm trong nhóm **dual-path BiMamba** (cùng DPMamba, SepMamba),
kế thừa khung Conformer (FFN + DepthwiseConv) và bổ sung **điều kiện hóa F0
thanh điệu** — điểm duy nhất không có ở bất kỳ mô hình nào trong bảng so sánh.
Dữ liệu từ DPMamba (22,6 dB @ 15,9 M) và SepMamba (22,7 dB @ 22 M) cho thấy
mức hiệu năng kỳ vọng ~21–23 dB SI-SNRi trên benchmark 8 kHz nếu được huấn
luyện tương đương, đồng thời chi phí MACs/bộ nhớ thấp hơn hẳn Sepformer
(258 G/s, 35 GB). Giá trị mới cần chứng minh bằng thực nghiệm nằm ở mức tăng
SI-SNRi trên tiếng Việt nhờ F0 — không mô hình nào trong nhóm này làm được.

---

## 8. Nguồn tham khảo

- [Dual-path Mamba (DPMamba), arXiv 2403.18257](https://arxiv.org/abs/2403.18257) — bảng so sánh DPMamba XS/S/M, S4M, QDPN, MossFormer(L), MossFormer2(L), TD-Conformer-XL.
- [SPMamba, arXiv 2404.02063](https://arxiv.org/abs/2404.02063) + [GitHub JusperLee/SPMamba](https://github.com/JusperLee/SPMamba) — 22,5 dB / 6,14 M / 238,69 G/s; bảng Conv-TasNet, DPRNN, BSRNN, TF-GridNet.
- [SepMamba, arXiv 2410.20997](https://arxiv.org/abs/2410.20997) + [GitHub andrasschin/SepMamba](https://github.com/andrasschin/SepMamba) — bảng so sánh đầy đủ (SepFormer, MossFormer S/M/L, MossFormer2, TF-GridNet S/M/L, SepMamba S/M) kèm MACs, latency, bộ nhớ.
- [MOSSFormer2, arXiv 2312.11825](https://arxiv.org/abs/2312.11825) — transformer + GFRN.
- [MOSSFormer (ICASSP 2024), GitHub alibabasglab/MossFormer](https://github.com/alibabasglab/MossFormer).
- [TF-GridNet, arXiv 2211.12433](https://arxiv.org/abs/2211.12433).
- [Mamba-TasNet / Speech Slytherin, arXiv 2407.09732](https://arxiv.org/abs/2407.09732) + [GitHub xi-j/Mamba-TasNet](https://github.com/xi-j/Mamba-TasNet).
- SepFormer (Subakan et al., ICASSP 2021); Conv-TasNet (Luo & Mesgarani, 2019); DPRNN (Luo et al., 2020).
