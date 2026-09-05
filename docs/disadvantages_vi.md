# Nhược điểm & Hạn chế của Mô hình Hybrid Conformer-BiMamba

*Đánh giá phê bình, thẳng thắn. Mọi con số dưới đây được đo bằng cách khởi tạo
các mô hình thực tế trong repository này với cấu hình mặc định (`configs/*.yaml`).*

---

## 0. Tóm tắt (TL;DR)

| # | Nhược điểm | Mức độ | Giảm thiểu |
|---|---|---|---|
| D1 | **Nhiều tham số hơn baseline MHSA** (+53%: 16,4 M so với 10,7 M) — luận điểm O(n) nói về *tỉ lệ tăng theo chiều dài chuỗi*, không phải kích thước mô hình | Cao | `expand=1`, `d_state=32`, bớt số khối, hoặc giảm `d_model` |
| D2 | Huấn luyện Mamba nhạy với learning rate / dễ mất ổn định | Cao | `mamba_lr_scale=0.5`, warmup 5 k bước, grad clip, khởi tạo `dt` thận trọng |
| D3 | Hai chiều = gấp đôi chi phí Mamba một chiều; gradient checkpointing làm chậm huấn luyện | Trung bình | Chỉ dùng khi thiếu bộ nhớ; đường nhanh `use_mamba2` |
| D4 | mamba-ssm chỉ chạy trên Linux+CUDA; không có kernel CPU nhanh; triển khai edge cần port tùy chỉnh | Cao | Fallback thuần PyTorch (chỉ để phát triển); port C++/ONNX cho sản xuất |
| D5 | Điều kiện hóa F0 chỉ tốt bằng chất lượng bộ ước lượng cao độ; F0 của tín hiệu trộn là khái niệm không rõ ràng với 2 người nói | Trung bình | Chuẩn hóa median, làm mượt stonemask, cổng tanh mềm, ablation không-F0 |
| D6 | Thiết kế phi nhân quả — không dùng được cho tách giọng streaming/online | Trung bình | Đã ghi rõ; biến thể nhân quả = một nửa mô hình |
| D7 | Chunking hai chiều (250/125) đã chặn phần bậc hai của attention → lợi ích O(n) có thật nhưng nhỏ hơn so với so sánh cả chuỗi | Thấp | Đo latency tường minh |
| D8 | Vivos nhỏ; SI-SNR không đo được độ đúng của thanh điệu | Trung bình | Bổ sung Common Voice; tùy chọn kiểm tra ASR |

---

## D1. Mô hình *lớn hơn* baseline — luận điểm độ phức tạp nói về tỉ lệ tăng, không phải kích thước

Số tham số đo được (cấu hình mặc định: `d_model=256`, `d_ffn=1024`, 4+4 khối):

| Mô hình | Tham số | So với baseline |
|---|---|---|
| Vanilla Conformer (MHSA) | **10.740.224 (10,74 M)** | — |
| Hybrid Bi-Mamba (không F0) | **16.294.400 (16,29 M)** | **+5,55 M (+51,7 %)** |
| Hybrid Bi-Mamba + F0 | **16.430.816 (16,43 M)** | **+5,69 M (+53,0 %)** |

Vì sao: trong mỗi khối, lớp con MHSA tốn 263.680 tham số (4·d² ≈ 0,26 M) trong
khi lớp con Bi-Mamba tốn **974.592** (hai lõi × ~487 k; mỗi lõi = `in_proj`
262 k + `out_proj` 131 k + `x_proj` 74 k + `dt_proj` 8,7 k + `f0_gate` 8,3 k +
`conv1d` 2,6 k + `D` 0,5 k). **Bi-Mamba ≈ 3,7× chi phí của lớp attention**; hai
chiều khiến nó tốn ~7,4× chỗ đó. Các FFN (1,05 M/khối) chiếm ưu thế ở cả hai
mô hình như nhau, nên việc thay thế thêm +710.912 mỗi khối × 8 khối ≈ +5,7 M.

Hệ quả:
- VRAM cao hơn với *cùng độ rộng* và nhiều FLOP hơn mỗi bước ở các phép chiếu
  (khoản tiết kiệm nằm ở ma trận attention mà phép quét thay thế).
- Cách diễn đạt trung thực: lợi thế của Mamba là **O(n) tính toán/bộ nhớ so với
  O(n²) của ma trận attention** khi chiều dài chuỗi tăng — *không phải* mạng
  nhỏ hơn.
- Để đạt ngang tham số với baseline, phải thu nhỏ bản hybrid (`expand=1`,
  `d_state=32`, hoặc ít khối hybrid hơn) — có thể làm giảm độ chính xác.
- Các mô hình lai SSM/Transformer trong tài liệu (ví dụ MOSSFormer cho tách
  giọng, Samba cho LM) thường dùng `d_model` *nhỏ hơn* cho nhánh Mamba chính
  vì lý do này.

## D2. Tính ổn định khi huấn luyện Mamba

- Các lớp SSM nổi tiếng là nhạy với learning rate: `dt` và `A` nằm trên thang
  log; LR quá lớn đẩy `softplus(dt + dt_bias)` ra ngoài dải hợp lệ và phép
  quét trở nên bùng nổ hoặc chết.
- Các biện pháp giảm thiểu đã cài đặt (và chúng là *chi phí*, không miễn phí):
  LR riêng 0,5× cho nhóm tham số SSM, warmup 5.000 bước, grad clip 5.0, khởi
  tạo `dt_proj` thận trọng. Những điều này làm hội tụ ban đầu chậm hơn baseline.
- AMP fp16 tương tác xấu với giá trị `dt` rất nhỏ (underflow) — kernel CUDA
  giảm thiểu điều này, nhưng mọi đường tùy chỉnh/CPU phải giữ `dt` ở fp32.

## D3. Hai chiều làm gấp đôi chi phí Mamba

- Hai lõi (fwd + bwd) = 2× tham số, 2× thời gian quét, 2× bộ nhớ activation so
  với Mamba một chiều. Vẫn rẻ hơn attention với chuỗi dài, nhưng hệ số là có
  thật.
- Gradient checkpointing (`checkpointing: true`) giảm bộ nhớ activation nhưng
  thêm chi phí tính lại (thường +20–40 % thời gian mỗi bước).

## D4. Ma sát về phụ thuộc và triển khai

- `mamba-ssm` chỉ biên dịch trên Linux với bộ công cụ CUDA (C++/CUDA/ninja);
  ràng buộc phiên bản với `causal-conv1d` và Triton là nỗi đau phổ biến.
- **Không có kernel CPU nhanh**: phép quét tham chiếu thuần PyTorch tuần tự
  theo thời gian và không phù hợp sản xuất. Đo trên máy Mac này (Apple
  M-series, mô hình đồ chơi 24-dim/2 khối): ≈ 158 ms cho 1 giây âm thanh — tức
  chậm hơn real-time ~6× với mô hình *tí hon*; mô hình 256-dim đầy đủ còn chậm
  hơn nhiều. Triển khai edge/CPU đòi hỏi port C++/ONNX/GGML của phép quét.
- Kernel hợp nhất `Mamba2` không nhận B, C từ ngoài, nên đường có điều kiện hóa
  F0 bị buộc dùng kernel Mamba-1 cũ hơn (chậm hơn, tốn bộ nhớ hơn).
- Trên máy không có CUDA, toàn bộ mô hình âm thầm chạy phép quét tham chiếu:
  đúng nhưng chậm — một cái bẫy tái lập kết quả cho đội không có đúng stack GPU.

## D5. Điểm yếu của điều kiện hóa F0

1. **F0 của tín hiệu trộn không được định nghĩa rõ.** Hai người nói đồng thời
   → hai đường cao độ; PyWorld chỉ trả về một. Proxy khi huấn luyện
   `max(f1, f2)` (cao độ trội) là xấp xỉ thô của những gì mô hình thấy khi suy
   luận (PyWorld trên tín hiệu trộn thật) — **lệch phân phối giữa huấn luyện
   và suy luận**.
2. **Lỗi bộ ước lượng cao độ**: lỗi octave, khung vô thanh, và phát âm kẹt
   (creaky) — tiếng Việt có thanh ngã/nặng thường kẹt → F0 yếu — làm hỏng tín
   hiệu điều kiện hóa.
3. **Thêm pipeline và latency**: khi suy luận, mô hình cần F0 của tín hiệu
   trộn *trước khi* tách → thêm một lượt PyWorld (~5–10× real-time trên CPU)
   hoặc một bộ ước lượng F0 học được.
4. **Cổng có thể bị bỏ qua.** Cổng là thang nhân điều biến tanh lên B, C; nếu
   mô hình học `gate ≈ 0`, điều kiện hóa âm thầm vô hiệu — phải so ablation 2
   và 3 (nếu SI-SNRi bằng nhau nghĩa là cổng đã sụp đổ).
5. **Dung lượng hạn chế**: `f0_dim=64 → 2·d_state=128` giá trị vô hướng mỗi lớp
   là kênh thô cho thông tin thanh điệu (chỉ 3 đặc trưng vô hướng mỗi khung).
6. **Rủi ro đường tắt nhận dạng người nói**: F0 tương quan với nhận dạng người
   nói; chuẩn hóa median loại bỏ mức tuyệt đối, nhưng rò rỉ còn lại có thể khiến
   mô hình khai thác *loại* người nói thay vì *đường nét* thanh điệu.

## D6. Thiết kế phi nhân quả

- Bi-Mamba nhìn toàn bộ câu — tốt cho tách giọng offline (kiểu WSJ0-2mix)
  nhưng **không streaming**: biến thể nhân quả chỉ giữ lõi thuận, làm mất một
  nửa mô hình và mất ngữ cảnh thanh điệu *từ cả hai phía* (đường nét thanh
  điệu thường thể hiện qua âm tiết theo sau — chính là thông tin hệ nhân quả
  mất đi).

## D7. Lợi ích độ phức tạp bị chặn bởi chunking hai chiều

- Attention được áp trên chiều chunk S=250 và C≈159, **không phải** chuỗi
  20 k khung. Nên ma trận attention là (250×250) và (159×159) mỗi head — bậc
  hai nhưng bị chặn (~318 M số thực cho intra tại B=4, fp32). Mamba vẫn thắng
  (state ≈ 83 MB so với attention ≈ 1,3 GB tại B=4) nhưng câu "O(n) vs O(n²)"
  chỉ đúng trọn vẹn khi chunk lớn hơn.
- Cửa sổ overlap-add (ràng buộc hop = chunk/2) có thể làm nhòe các khởi phát
  (onset) sắc nét; chất lượng tái tạo phụ thuộc cửa sổ tam giác.
- Thêm siêu tham số cần tinh chỉnh (chunk size, hop, cửa sổ) mà baseline cũng
  có, nhưng chúng tương tác với hiệu quả quét.

## D8. Dữ liệu, đánh giá và ý nghĩa thống kê

- **Vivos nhỏ** (~15 h huấn luyện). Dù có Common Voice + dynamic mixing, mô
  hình có thể quá khớp ~65 người nói huấn luyện; chênh lệch SI-SNRi ±0,5 dB
  giữa các ablation có thể không đạt ý nghĩa thống kê — cần kiểm định ghép cặp
  trên nhiều hỗn hợp giữ riêng.
- **SI-SNRi không đo độ đúng của thanh điệu**: hai đầu ra có thể cùng điểm
  SI-SNRi nhưng một cái giữ thanh điệu, một cái không. PESQ/STOI giúp ích
  nhưng không nhạy thanh điệu; kiểm tra ASR trên luồng đã tách là phép thử thật.
- So sánh benchmark (Sepformer/TF-GridNet) được huấn luyện trên dữ liệu khác
  (WSJ0-2mix, 8 kHz, 30 h) — số liệu không chuyển đổi trực tiếp sang hỗn hợp
  tiếng Việt 32 kHz.
- Lợi ích F0 (nếu có) phụ thuộc ngôn ngữ: việc *thay Mamba* chuyển được sang
  mọi ngôn ngữ; *điều kiện hóa F0* được tinh chỉnh cho tiếng Việt thanh điệu
  và có thể không chuyển sang ngôn ngữ không thanh điệu.

---

## Thứ tự đọc đề xuất
1. [`parameter_analysis_vi.md`](parameter_analysis_vi.md) — toán tham số chính xác
   và dự đoán kết quả.
2. [`report.md`](report.md) — chi tiết kiến trúc (tiếng Việt).
3. [`README.md`](../README.md) — cách dùng, cấu hình, bảng rủi ro.
