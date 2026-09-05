# Báo cáo: Tích hợp FSMN (GFRN) của MOSSFormer2 vào Hybrid Conformer-BiMamba

*Đánh giá phương án, ưu/nhược điểm và độ khả quan khi bổ sung mô-đun
Gated-FSMN / GFRN (cơ chế "hồi quy không-RNN" của MOSSFormer2) vào mô hình
Hybrid Conformer-BiMamba vừa viết lại trong framework ClearerVoice-Studio
(`train/speech_separation/`). Mọi số tham số trong báo cáo đều được **đo
thực tế** từ code đã vendor.*

---

## 1. FSMN / GFRN trong MOSSFormer2 thực chất là gì?

Từ code thật trong `models/mossformer2/` (đã vendor vào framework):

| Thành phần | Vai trò |
|---|---|
| `UniDeepFsmn` | FSMN gốc: `Linear→ReLU→project` rồi **conv 2D theo trục thời gian** với kernel rộng `2·lorder−1` (=39 khung khi `lorder=20`), **groups = output_dim** (depthwise), có residual `input + out`. Đây là "bộ nhớ" cửa sổ — nắm mẫu hồi quy cục bộ mà không dùng vòng lặp RNN. |
| `UniDeepFsmn_dilated` + `DilatedDenseNet` | Bản nâng cấp: chuỗi conv **dilated** (dilation 1,2,4,…) với **kết nối dense** (concat skip), `InstanceNorm2d(affine)` + `PReLU` → mở rộng receptive field theo thời gian. |
| `Gated_FSMN(_dilated)` | Bọc GLU: hai nhánh `to_u`, `to_v` (FFConvM) + FSMN → cổng `u * sigmoid(v)` — kiểm soát luồng thông tin. |
| `Gated_FSMN_Block(_Dilated)` | Khối hoàn chỉnh: `conv1 (D→inner) + PReLU → CLayerNorm → Gated_FSMN → CLayerNorm → conv2 (inner→D) + residual`. **Số đo: 0,44M tham số/khối @ D=256** (conv1 66k + gated_fsmn 304k + conv2 66k). |

**Vị trí trong MOSSFormer2:** mỗi lớp của `MossformerBlock_GFSMN` chạy
`attention local-global (FLASH_ShareA_FFConvM) → Gated_FSMN_Block_Dilated`,
lặp `depth` lần — tức MOSSFormer2 = **attention + GFRN nối tiếp nhau**, GFRN
đảm nhiệm nốt phần "mẫu hồi quy tầm mịn" (fine-scale recurrent patterns) mà
attention không bắt tốt. Đây chính là phần đóng góp đưa MOSSFormer2 lên
24,1 dB trên WSJ0-2Mix (vs MOSSFormer 22,8 dB).

Quan trọng: `Gated_FSMN_Block_Dilated` nhận tensor `(B, L, D)` và trả về
`(B, L, D)` — **khớp sẵn với định dạng token `(B, S, N)` / `(B, C, N)` mà
các block của ta đang dùng**, nên có thể chèn trực tiếp không cần đổi kiểu
dữ liệu.

---

## 2. Cách tích hợp vào Hybrid Conformer-BiMamba

### Khối hiện tại (đã kiểm thử)

```
HybridBlock:  x → +0.5·FFN1 → +BiMamba → +DWConv(k=31) → +0.5·FFN2 → x
              (PreNorm + residual ở từng lớp con; BiMamba có F0 gate trên B, C)
```

### Phương án A (khuyến nghị) — nối tiếp sau BiMamba, kiểu MOSSFormer2

```
HybridBlock v2:  x → +0.5·FFN1 → +BiMamba → +GFSMN → +DWConv → +0.5·FFN2 → x
```

- Giống hệt triết lý MOSSFormer2 (attention → GFRN): Bi-Mamba lo "phụ thuộc
  chọn lọc dài hạn", GFSMN lo "mẫu cục bộ/recurrent-like tầm mịn".
- Thêm `PreNorm + residual` quanh GFSMN như các lớp con khác (GFSMN đã có
  residual trong, nhưng bọc ngoài vẫn nhất quán với khối).

```python
# models/hybrid_bimamba/hybrid_block.py (bổ sung)
from models.mossformer2.mossformer2_block import Gated_FSMN_Block_Dilated

class HybridBlock(nn.Module):
    def __init__(self, d_model, ..., use_fsmn=False, fsmn_inner=None):
        ...
        if use_fsmn:
            self.fsmn = PreNorm(d_model, Gated_FSMN_Block_Dilated(
                dim=d_model, inner_channels=fsmn_inner or d_model))
    def forward(self, x, f0=None):
        x = x + 0.5 * self.ffn1(x)
        x = x + self.mamba(x, f0)
        if hasattr(self, "fsmn"):
            x = x + self.fsmn(x)          # ← GFSMN nối tiếp sau BiMamba
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return x
```

Vì code MOSSFormer2 đã nằm sẵn trong framework (`models/mossformer2/fsmn.py`,
`conv_module.py`, `mossformer2_block.py`), chỉ cần **import + 1 cờ config**,
không cần copy lại.

### Phương án A′ — song song với BiMamba (nhánh phụ)

`x = x + self.mamba(x, f0) + self.fsmn(x)` — giảm độ sâu nối tiếp, tăng
biểu diễn ngang; ít rủi ro "chuỗi quá sâu", nhưng khác kiến trúc MOSSFormer2
(nối tiếp) nên khó đối chiếu trực tiếp.

### Phương án B — chỉ đặt GFSMN ở một đường

Ví dụ: GFSMN **chỉ ở đường inter (liên đoạn)** hoặc chỉ 1–2 khối đầu/cuối —
giảm tham số, dùng làm ablation xem GFSMN giúp ích ở mức ngữ cảnh nào.

### Phương án C — thay DepthwiseConv bằng GFSMN

Không khuyến nghị: mất sự so sánh sạch với baseline Conformer (DWConv là
"bias cục bộ" dùng chung cho cả hai khối trong thiết kế ablation).

### Cấu hình & tham số dự kiến

| Cấu hình | Khối lõi | Tham số (mô hình M, đo thật) |
|---|---|---|
| Conformer-M (baseline) | MHSA | 10,74 M |
| Hybrid-F0-M (hiện tại) | BiMamba + F0 | 16,43 M |
| **Hybrid-F0+FSMN-M (đề xuất)** | BiMamba + F0 + GFSMN | **≈ 19,9 M** (+0,44 M/khối × 8 khối ≈ +3,5 M, **+21 %**) |

Cờ mới cần thêm: `use_fsmn`, `fsmn_inner` (mặc định = d_model), `fsmn_lorder`
(mặc định 20), thêm vào `train.py` / `inference.py` / yaml (giữ đúng quy ước
tên arg của framework). F0 gate: **giữ nguyên cho BiMamba**; nếu muốn, sau này
có thể tái dùng vector F0 để gate thêm nhánh FSMN (tốn thêm 1 Linear nhỏ).

---

## 3. Ưu điểm

1. **Bổ trợ đúng chỗ Mamba còn thiếu**: Mamba dùng conv cục bộ ngắn
   (`d_conv=4`) + trạng thái chọn lọc; GFSMN cung cấp **bộ nhớ cửa sổ 39 khung
   + dense dilated** với cổng GLU — bắt "mẫu lặp tầm mịn" (micro-prosody, phụ
   âm, trường hợp **creaky của thanh ngã/nặng tiếng Việt**) mà trạng thái
   Mamba học gián tiếp. Với thanh điệu: contour dài (~60–300 ms) do
   BiMamba+F0 gate đảm nhiệm; chi tiết ngắn do GFSMN — phân công bổ sung hợp lý.
2. **Bằng chứng mạnh từ MOSSFormer2**: GFRN là phần đưa 22,8 → 24,1 dB trên
   WSJ0-2Mix; đây là "recurrent module" đã kiểm chứng, không phải ý tưởng mù.
3. **Chi phí cài đặt rất thấp**: code đã vendor sẵn trong framework, khớp
   `(B,L,D)`, chỉ cần cờ config → rủi ro kỹ thuật nhỏ, làm được trong 1–2 ngày.
4. **Giữ triết lý RNN-free**: GFSMN song song hóa toàn chuỗi, cùng hướng với
   Mamba → độ phức tạp tổng thể vẫn thấp hơn nhiều so với transformer thuần.
5. **So sánh đối đầu trực tiếp**: với cùng framework/dataset, mô hình mới
   ("MOSSFormer2 nhưng thay attention bằng BiMamba + F0") có thể đối chiếu
   công bằng với chính MOSSFormer2 — tăng giá trị trình bày.

---

## 4. Nhược điểm & rủi ro

1. **Tham số +21 % (~3,5 M)** và thời gian/bước tăng tương ứng — mô hình vốn
   đã nặng hơn Conformer 53 %; cần kiểm soát (giảm `d_state` hoặc bớt khối khi
   bật FSMN để so sánh cùng tham số).
2. **Trùng lặp chức năng tiềm ẩn**: DWConv (31 taps) + `d_conv` của Mamba +
   GFSMN đều là bộ lọc cục bộ → lợi ích biên có thể nhỏ (đặc biệt nếu FSMN
   chỉ học lại điều DWConv/Mamba đã làm); nguy cơ lợi ích ≈ 0 hoặc âm khi dữ
   liệu nhỏ.
3. **Chuỗi khối sâu hơn** (5 lớp con/khối) → chậm hội tụ, nhạy LR hơn; việc
   khóa "mamba_lr_scale" kiểu cũ không áp cho FSMN — cần nhóm LR riêng hoặc
   chấp nhận LR chung.
4. **F0 không điều kiện hóa FSMN**: BiMamba được gate thanh điệu còn GFSMN
   thì không → không nhất quán điều kiện hóa; muốn nhất quán phải thêm gate
   F0 cho FSMN (thêm tham số + code).
5. **Loãng câu chuyện novelty**: "Conformer-BiMamba + FSMN" về bản chất là
   **MOSSFormer2 lai thêm Mamba** — nên định vị là *biến thể mở rộng* để so
   sánh, không phải đóng góp chính; nếu viết báo, cần ablation tách bạch.
6. **Bộ nhớ/activation** tăng; trên Vivos nhỏ (~15 h) nguy cơ quá khớp tăng
   rõ — gain giả tạo hoặc âm khi đánh giá trên tập mới.

---

## 5. Độ khả quan (đánh giá ước lượng)

| Tiêu chí | Đánh giá | Lý do |
|---|---|---|
| Độ khó triển khai | **Thấp–Trung bình** | Code có sẵn trong framework; 1–2 ngày cho bản chạy được (A), +1 ngày ablation |
| Khả năng SI-SNRi tăng (dữ liệu lớn, WSJ0-class) | **Trung bình–Khá** | MOSSFormer2 cho +1,3 dB nhờ GFRN; kỳ vọng **+0,3–1,0 dB** khi cộng vào BiMamba nếu không trùng chức năng |
| Khả năng tăng trên Vivos nhỏ | **Thấp–Trung bình** | Dữ liệu ít → lợi ích của mô-đun thêm tham số dễ bị quá khớp nuốt mất; khuyến nghị chỉ bật FSMN ở 1–2 khối hoặc giảm tham số chỗ khác |
| Rủi ro regression | Thấp (bản thân nó không phá huỷ) nhưng lợi ích có thể ≈ 0 | Cần ablation đối chứng |
| Giá trị trình bày (paper) | Trung bình | Mạnh nếu gain dương trên tiếng Việt (chưa ai làm Mamba+GFRN+F0); yếu nếu ≈ 0 |

**Ước tính tổng thể: ~50–60 % khả năng có lợi ích ròng ≥ +0,3 dB trên dữ liệu
đủ lớn; ~30 % trên Vivos nhỏ.** Lợi ích lớn nhất kỳ vọng ở việc bắt
**micro-prosody/creaky** và cải thiện PESQ/STOI hơn là SI-SNRi thô.

### Kịch bản thí nghiệm đề xuất (cùng scp, cùng seed — framework đã đảm bảo)

| Chạy | Cấu hình | Mục đích |
|---|---|---|
| 1 | `HybridBiMamba_SS_16K` + F0 (hiện tại) | baseline mới |
| 2 | 1 + `use_fsmn: 1` **chỉ intra** | GFSMN giúp ở ngữ cảnh ngắn? |
| 3 | 1 + FSMN **chỉ inter** | GFSMN giúp ở ngữ cảnh dài? |
| 4 | 1 + FSMN **cả hai đường** (phương án A) | đầy đủ, so sánh với 2/3 |
| 5 | (đối chứng) `MossFormer2_SS_16K` gốc | xác lập mốc SOTA cùng dữ liệu |
| 6 | (tùy chọn) 4 nhưng giảm `d_state`/bớt khối để **cùng tham số** với 1 | tách bạch "lợi ích do thêm mô-đun" vs "do thêm tham số" |

Thời lượng ước tính: 4–6 ngày GPU (mỗi chạy M ~1,5–3,5 ngày trên 3090; có thể
dùng bản S để khảo sát trước ~1 ngày). Ngưỡng quyết định: nếu chạy 2/3/4 đều
không vượt baseline ≥ +0,3 dB (hoặc chỉ đạt khi tăng tham số), nên **không
đưa FSMN vào mô hình chính** — giữ thiết kế sạch hiện tại, dùng FSMN chỉ như
ablation đối chứng trong báo cáo.

---

## 6. Kết luận & khuyến nghị

- **Về kỹ thuật**: khả thi cao — GFSMN (0,44 M/khối) chèn trực tiếp vào
  `HybridBlock` với 1 cờ config, tận dụng code đã vendor; tổng mô hình ~19,9 M.
- **Về giá trị**: trung bình — lợi ích kỳ vọng +0,3–1,0 dB trên dữ liệu lớn,
  thấp trên Vivos; rủi ro chính là trùng chức năng cục bộ với DWConv/`d_conv`
  và quá khớp do tham số tăng.
- **Khuyến nghị hành động**: (1) giữ mô hình hiện tại làm baseline chính;
  (2) chạy kịch bản 2–4 ở bản S trước để đo tín hiệu trong ~1–2 ngày; (3) chỉ
  đưa FSMN vào bản chính nếu gain ≥ +0,3 dB sau khi đã cân bằng tham số;
  (4) nếu giữ FSMN, cân nhắc thêm F0 gate cho nhánh FSMN để giữ nhất quán
  điều kiện hóa thanh điệu.
