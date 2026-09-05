# Báo cáo kiến trúc chi tiết: Hybrid Conformer-BiMamba cho Tách Giọng Nói Tiếng Việt

**Phiên bản:** 1.0 · **Trạng thái:** đã cài đặt và kiểm thử (forward/backward hợp lệ trên CPU/MPS; huấn luyện cần GPU CUDA để dùng kernel Mamba tối ưu)

---

## 1. Giới thiệu và bối cảnh

Tiếng Việt là ngôn ngữ **có thanh điệu**: sáu thanh (ngang, huyền, sắc, hỏi, ngã, nặng) được thể hiện chủ yếu qua **đường nét (contour) của tần số cơ bản F0** trong cửa sổ khoảng 60–300 ms. Khi hai người nói có cao độ chồng lấn, đường nét F0 là một trong số ít manh mối để phân biệt họ về mặt âm vị học.

Các mô hình tách giọng nói hiện đại (Sepformer, TF-GridNet) dựa trên **Multi-Head Self-Attention (MHSA)** — có độ phức tạp **bậc hai O(n²)** theo chiều dài chuỗi và chi phí bộ nhớ lớn. Mô hình đề xuất **Hybrid Conformer-BiMamba**:

1. **Thay MHSA bằng Bidirectional Mamba (Bi-Mamba)** trong khối Conformer → quét trạng thái **tuyến tính O(n)**, vẫn nắm được phụ thuộc xa.
2. **Điều kiện hóa (conditioning) thanh điệu**: chèn vector nhúng F0 của tín hiệu trộn làm **cổng điều biến thích ứng (adaptive scaling)** lên hai ma trận chiếu **B, C** của khối Mamba → quá trình chuyển trạng thái phụ thuộc vào ngữ cảnh thanh điệu tức thời.
3. **Giữ nguyên Depthwise Convolution** (bias cục bộ <30 ms cho âm vị/phụ âm bật) và **giữ nguyên FFN** — chỉ riêng việc đổi MHSA → Bi-Mamba mới tạo khác biệt về hiệu năng, giúp phép thử loại trừ (ablation) sạch.

---

## 2. Kiến trúc tổng thể

```
mixture (B, 1, T)  @ 32 kHz
   │
   ├─ Encoder: Conv1d(kernel=16, stride=8) + chuẩn hóa toàn cục ─► E (B, 256, T')
   │
   ├─ [Nhánh F0] F0 thô (Hz, khung 10 ms)
   │      └─ resample_f0 ─► F0Projector ─► f0_emb (B, T', 64)
   │
   └─ Xử lý hai chiều (dual-path, kiểu Sepformer)                E (B, 256, T')
        │
        ├─ chunk: T' ──(S=250, hop=125)──► (B, C, S, 256)  + PE hình sin (chiều S)
        │      └─ chồng intra: [FFN → BiMamba → DWConv → FFN] × 4  trên S
        │             BiMamba nhận F0 theo từng vị trí  (B·C, S, 64)
        ├─ inter: chuyển vị → (B, S, C, 256)  + PE hình sin (chiều C)
        │      └─ chồng inter: cùng loại khối × 4  trên C
        │             BiMamba nhận tóm tắt F0 mỗi chunk (B·S, C, 64)
        └─ overlap-add (cửa sổ tam giác) ──► H (B, 256, T')
   │
   ├─ masker: Linear(256, 512) → sigmoid → m1, m2
   └─ decoder: ConvTranspose1d(kernel=16, stride=8) ──► est1, est2 (B, 2, 1, T_dec)
```

Đầu vào miền thời gian → mã hóa học (learned encoder) → tách thành các đoạn (chunk) → xử lý **trong đoạn** (intra) và **liên đoạn** (inter) bằng các khối hybrid → ghép lại bằng overlap-add → ước lượng mặt nạ (mask) cho 2 người nói → giải mã về dạng sóng.

---

## 3. Các khối thành phần chi tiết

### 3.1 Encoder / Decoder

| Thành phần | Tham số | Vai trò |
|---|---|---|
| Encoder | `Conv1d(1→256, k=16, s=8)` + chuẩn hóa `x/√(mean(x²)+ε)` | Trích đặc trưng dạng sóng; mỗi khung ~4 ms tại 32 kHz |
| Decoder | `ConvTranspose1d(256→1, k=16, s=8)` | Tái tạo dạng sóng từ đặc trưng đã được mặt nạ |

### 3.2 Xử lý hai chiều (Dual-Path)

- Chunk: `T' → (B, C, S, 256)` với `S = chunk_size = 250`, `hop = 125` (chồng lấn 50%).
- **Chồng intra** áp dụng khối hybrid trên chiều `S` (ngữ cảnh trong một đoạn ~62,5 ms — vừa đủ cho đường nét thanh điệu).
- **Chồng inter** áp dụng trên chiều `C` (ngữ cảnh toàn câu).
- Overlap-add dùng **cửa sổ tam giác** (độ dài `2·hop = S`) đảm bảo tổng các cửa sổ bằng 1 tại mọi điểm, tránh nhiễu biên giữa các đoạn.
- **Positional Encoding hình sin** được cộng vào cả hai chiều S và C — Mamba nhạy thứ tự nhưng không có cơ chế vị trí tường minh như Transformer, nên PE là bắt buộc (khớp yêu cầu "input đã chứa positional encoding trước khi vào khối Conformer").

### 3.3 Khối Hybrid (thay thế trực tiếp MHSA)

```
Khối Conformer (baseline):  x → 0.5·FFN → +MHSA → +DWConv → 0.5·FFN → x
Khối Hybrid (đề xuất):      x → 0.5·FFN → +BiMamba → +DWConv → 0.5·FFN → x
```

- Cả hai khối dùng **PreNorm + residual** đúng chuẩn Conformer (Gulati et al., 2020).
- **FFN** (expansion 4×, SiLU) và **DepthwiseConv1d** (kernel 31, groups = d_model, BatchNorm + SiLU, padding "same") **giống hệt nhau** ở cả hai khối → mọi chênh lệch hiệu năng đều quy về việc đổi MHSA → Bi-Mamba.
- Đầu ra Bi-Mamba được **cộng residual vào đầu vào**, sau đó qua LayerNorm (PreNorm kế tiếp) trước khi vào Depthwise Conv — đúng hợp đồng residual/chuẩn hóa của Conformer.

### 3.4 Lớp Bi-Mamba

**Lựa chọn triển khai (Option A):** hai lõi SSM song song.

```
h_fwd = SSM_fwd(x, f0)                      # quét nhân quả trái → phải
h_bwd = SSM_bwd(flip(x), flip(f0))          # quét nhân quả trong chiều ngược
y     = h_fwd + flip(h_bwd)                 # kết hợp: tổng | nối + Linear
```

- **Không rò rỉ dữ liệu tương lai**: mỗi chiều nhân quả *trong định hướng của nó*; việc kết hợp chỉ cho phép mô hình nhìn toàn bộ tín hiệu đầu vào — đúng thiết lập phi nhân quả của tách giọng nói offline (chuẩn WSJ0-2mix). Không dùng thông tin tương lai kiểu teacher-forcing.
- **Option B (cờ `bidirectional` trong mamba-ssm 2.x): KHÔNG tồn tại** — đã đối chiếu mã nguồn upstream (cả `Mamba` lẫn `Mamba2` đều không có tham số này). Nếu bản tương lai bổ sung, chỉ cần bật cờ trong `BiMambaLayer`.

**Lõi SSM (công thức Mamba-1 / S6):**

Lõi `F0ConditionedSSM` thực hiện biến đổi trạng thái rời rạc hóa (S4):

```
x_inner = SiLU(DepthwiseConv1d(x, d_conv=4))          # trộn cục bộ trước khi quét
Δ       = softplus(Δ_proj(x) + Δ_bias)                # bước thời gian phụ thuộc đầu vào
A       = −exp(A_log)                                 # ma trận chuyển trạng thái (d_inner, d_state)
B       = B_proj(x) · (1 + tanh(gate_B(f0)))          # ← điều biến F0
C       = C_proj(x) · (1 + tanh(gate_C(f0)))          # ← điều biến F0
Ā_t     = exp(Δ_t · A)                                # rời rạc hóa
B̄_t     = Δ_t · B_t
h_t     = Ā_t · h_{t−1} + B̄_t · x_t                   # vòng lặp trạng thái
y_t     = C_t · h_t + D · x_t
y       = y · SiLU(z)                                 # cổng đầu ra (gated output)
out     = out_proj(y)
```

- `d_state = 64` mặc định; độ phức tạp **O(n)** thay vì O(n²) của attention.
- Trên GPU: dùng kernel hợp nhất `mamba_ssm.ops.selective_scan_interface.selective_scan_fn` — kernel này **nhận B, C tường minh**, chính là điểm chèn F0.
- Trên máy không có mamba-ssm (macOS/CPU/CI): tự động chuyển sang **quét tham chiếu thuần PyTorch** (chậm, chỉ để phát triển/suy luận nhỏ).
- `use_mamba2=true` (chỉ khi không điều kiện hóa F0): dùng `Mamba2` hợp nhất (Triton) — nhanh hơn ~2–3×, tiết kiệm bộ nhớ; không dùng được với F0 vì B, C nằm bên trong kernel.

### 3.5 Mô-đun điều kiện hóa F0 (đóng góp riêng cho tiếng Việt)

1. **Trích xuất offline** (`preprocess/extract_f0.py`): PyWorld `dio` + `stonemask` (làm mượt) trên *tín hiệu trộn*, khung 10 ms (100 fps), dải 60–400 Hz, `0 = vô thanh (unvoiced)`. Lưu `.npy` để huấn luyện không phải trả chi phí pyworld online.
2. **`F0Projector`** (`models/f0_conditioning.py`): với mỗi khung, đặc trưng `[log-F0, cờ vô thanh, log-F0 − median(log-F0)]` → MLP 3→32→32→64. **Chuẩn hóa median** loại bỏ thành phần nhận dạng người nói (cao độ tuyệt đối), giữ lại **đường nét** — vật mang thanh điệu.
3. **Cổng điều biến trong lõi SSM** (`models/ssm.py`): mỗi lớp có cổng riêng `Linear(f0_dim→2·d_state) + tanh`, áp dụng **thang nhân thích ứng**:
   ```
   gate_B, gate_C = f0_gate(f0_emb).chunk(2, -1)      # tanh ∈ (−1, 1)
   B ← B · (1 + gate_B)
   C ← C · (1 + gate_C)
   ```
   → ma trận B, C phụ thuộc vào ngữ cảnh thanh điệu tức thời của người nói trong hỗn hợp. Khi `f0_dim = 0` cổng bị vô hiệu (ablation 2).
4. **F0 khi huấn luyện (dynamic mixing):** không thể biết F0 của tín hiệu trộn trước, nên dùng **`max(f1, f2)`** (cao độ trội) làm xấp xỉ. **Val/test dùng F0 của tín hiệu trộn thật** — đúng những gì mô hình nhìn thấy khi suy luận. Sai lệch này được đo bằng ablation 2 (không F0).

### 3.6 Masker và hàm mất mát

- `Linear(256, 512)` trên đặc trưng đã ghép lại → tách 2 mặt nạ → `sigmoid` → `est_i = mask_i · E` → decoder.
- Hàm mất mát: **SI-SNR với Permutation Invariant Training (PIT)** cấp câu cho 2 người nói — mô hình tự chọn phép gán (identity/swap) tối ưu; được kiểm thử: ước lượng = tín hiệu trộn ⇒ SI-SNRi = 0,0 dB; tách hoàn hảo ⇒ +26 dB; hoán vị vẫn cho cùng kết quả.

---

## 4. Điểm nâng cấp chính (so với baseline và mô hình cũ)

| # | Điểm nâng cấp | Baseline (Conformer/Sepformer) | Mô hình đề xuất | Lợi ích |
|---|---|---|---|---|
| 1 | **Độ phức tạp chuỗi** | MHSA: **O(n²)** thời gian và bộ nhớ | Bi-Mamba: **O(n)** tuyến tính | Xử lý câu 4–5 s @ 32 kHz (~20 k khung) với chi phí tuyến tính; suy luận nhanh hơn trên A100 và CPU |
| 2 | **Điều kiện hóa thanh điệu (F0)** | Không có; mô hình phải học mù thanh điệu | Cổng tanh thích ứng lên **B, C** của SSM, dẫn F0 từ tín hiệu trộn | Phân biệt người nói có cao độ chồng lấn; nhạy với đường nét thanh điệu tiếng Việt |
| 3 | **Bias cục bộ** | DepthwiseConv (giữ nguyên) | DepthwiseConv (giữ nguyên) | Không đánh mất đặc trưng âm vị/phụ âm bật <30 ms |
| 4 | **Bộ nhớ huấn luyện** | Attention lưu ma trận (B, n_head, L, L) | Trạng thái (B, d_inner, d_state) nhỏ cố định | Dùng được batch lớn hơn trên GPU 24 GB; có thể bật gradient checkpointing nếu cần |
| 5 | **Tốc độ suy luận / biên (edge)** | Attention bậc hai, khó nhúng | Quét tuyến tính; lõi thuần PyTorch chạy được CPU (tham chiếu) | Phù hợp thiết bị biên; latency đo được bằng `evaluate.py --cpu-latency` |
| 6 | **Ablation sạch** | — | Khối baseline và hybrid dùng chung FFN + DWConv, chỉ khác MHSA↔BiMamba | Mọi chênh lệch SI-SNRi quy về đúng phép thay thế |

### 4.1 Tại sao Mamba giữ được phụ thuộc xa?

MHSA lý tưởng nhưng O(n²). Mamba là SSM chọn lọc: ma trận chuyển trạng thái `A` và các chiếu `B, C` **phụ thuộc đầu vào** (`input-dependent`), nên nó có thể "chọn" giữ hay quên thông tin — mô phỏng cơ chế chú ý mà không cần bảng attention tường minh. Với thanh điệu tiếng Việt (contour kéo dài vài trăm ms), trạng thái `h_t` đóng vai trò "bộ nhớ" mang đường nét F0 dọc câu — chính là lý do điều biến B, C theo F0 có ý nghĩa.

### 4.2 Tại sao điều kiện hóa B, C (chứ không phải A)?

- `A` quyết định tốc độ suy giảm trạng thái (thang thời gian) — thay đổi nó dễ gây mất ổn định.
- `B, C` là "cổng vào/ra" của thông tin: điều biến chúng cho phép mô hình **tăng/giảm mức độ đưa ngữ cảnh thanh điệu vào trạng thái và đọc nó ra** — cách tác động mềm, ổn định (tanh ∈ (−1,1)), sai số F0 (khung vô thanh, lỗi octave) chỉ làm suy giảm nhẹ thay vì phá vỡ.

---

## 5. So sánh với các mô hình hiện có

| Tiêu chí | Vanilla Conformer | Sepformer | TF-GridNet | **Hybrid Conformer-BiMamba (đề xuất)** |
|---|---|---|---|---|
| Cơ chế phụ thuộc xa | MHSA O(n²) | Transformer dual-path O(n²) | Gated DPRNN + attention | **Bi-Mamba dual-path O(n)** |
| Nhận biết thanh điệu | Không | Không | Không | **Có (F0 gate trên B, C)** |
| Bias cục bộ | DepthwiseConv | DepthwiseConv | Conv | DepthwiseConv (giữ nguyên) |
| Bộ nhớ | Cao (attention) | Cao | Rất cao | **Thấp (trạng thái h nhỏ)** |
| Độ phức tạp huấn luyện | Trung bình | Trung bình | Cao | Trung bình (SSM cần LR thấp hơn) |
| Dữ liệu tiếng Việt | Cần lượng lớn | Cần lượng lớn | Cần lượng lớn | **Tận dụng F0 — phù hợp dữ liệu nhỏ (Vivos ~15 h)** |

Lưu ý: Sepformer/TF-GridNet là các điểm tham chiếu (ablation 4 tùy ngân sách tính toán), không phải thay thế trong phạm vi này.

---

## 6. Chi tiết cài đặt và các đường kernel

| Đường thực thi | Điều kiện | Ghi chú |
|---|---|---|
| `selective_scan_fn` (mamba-ssm CUDA) | GPU Linux + cài mamba-ssm | Kernel Mamba-1; nhận B, C tường minh → điểm chèn F0 |
| `Mamba2` (Triton hợp nhất) | `use_mamba2=true`, không F0 | Nhanh nhất; B/C nằm trong kernel nên chỉ dùng ablation không điều kiện hóa |
| `selective_scan_ref` (thuần PyTorch) | macOS / CPU / CI | Quét tuần tự, không phụ thuộc thư viện; đủ cho phát triển, không cho tốc độ huấn luyện |

Các file chính: `models/ssm.py` (lõi + cổng F0 + quét tham chiếu + Mamba2Core), `models/bimamba.py` (hai chiều), `models/hybrid_block.py` (hai khối), `models/f0_conditioning.py`, `models/separation.py` (toàn mô hình).

---

## 7. Cấu hình mặc định và ước lượng tham số

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| encoder_dim | 256 | kernel 16, stride 8 |
| chunk_size / hop | 250 / 125 | cửa sổ ~62,5 ms, chồng 50% |
| num_intra / num_inter | 4 / 4 | khối hybrid cả hai chiều |
| d_ffn / conv_kernel | 1024 / 31 | FFN ×4; DWConv <30 ms |
| d_state / d_conv / expand | 64 / 4 / 2 | lõi SSM |
| f0_dim | 64 | nhúng F0 dùng chung toàn mô hình |
| combine | sum | nối + Linear tùy chọn |
| **Ước lượng tham số** | **≈ 10 M** | Sepformer-base ~2,6 M; lớn ~26 M |

Bộ nhớ: batch 4 × câu 4–5 s @ 32 kHz trong 24 GB VRAM với AMP fp16 là thoải mái; bật `model.checkpointing: true` nếu cần giảm thêm.

---

## 8. Hạn chế và công việc tương lai

1. **Xấp xỉ F0 khi huấn luyện**: `max(f1, f2)` chưa bằng F0 thật của tín hiệu trộn → có thể dùng CREPE ước lượng F0 trộn online (tốn tính toán) hoặc huấn luyện thêm bộ ước lượng F0 phụ.
2. **mamba-ssm yêu cầu Linux + CUDA**; bản thuần PyTorch chỉ để phát triển → cần C++ port cho triển khai edge tốc độ cao.
3. **Mamba-1 core dùng kernel cũ**: nâng lên Mamba-2 có điều kiện hóa B/C đòi hỏi viết lại kernel (SSD scan) — công việc tương lai rõ ràng.
4. **Đánh giá thanh điệu**: SI-SNR không đo độ chính xác thanh điệu; đề xuất thêm bài kiểm tra phụ trợ (ASR tiếng Việt trên đầu ra tách — ngoài phạm vi hiện tại).
5. **Cờ `bidirectional` của mamba-ssm**: theo dõi bản phát hành; nếu có, thay thế Option A.

---

## 9. Tài liệu tham khảo

- Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (2023)
- Dao & Gu, *Transformers are SSMs* (Mamba-2, 2024)
- Gulati et al., *Conformer: Convolution-augmented Transformer for Speech Recognition* (2020)
- Subakan et al., *Attention is All You Need In Speech Separation* (Sepformer, 2021)
- Luo & Mesgarani, *Conv-TasNet* (2018); Wang et al., *TF-GridNet* (2023)
- VIVOS corpus (NTU); Common Voice (Mozilla, tiếng Việt)
- Morise et al., *PyWorld / D4C* (trích xuất F0)

---

*Báo cáo tương ứng 1:1 với mã nguồn trong repository: `conformer_bimamba/models/`, `conformer_bimamba/data/`, `conformer_bimamba/preprocess/`, `train.py`, `evaluate.py`, `inference.py`.*
