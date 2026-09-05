# Hybrid Conformer-BiMamba — Tách giọng nói tiếng Việt (có điều kiện hóa F0/thanh điệu)

Mô hình tách 2-người-nói miền thời gian cho tiếng Việt: thay **Multi-Head
Self-Attention (MHSA)** trong khối Conformer bằng **Bidirectional Mamba
(Bi-Mamba)** (`FFN → Bi-Mamba → DepthwiseConv → FFN`), và **điều kiện hóa
thanh điệu**: đường nét F0 của tín hiệu trộn điều biến ma trận B, C của SSM.

```
Conformer (baseline):  FFN → MHSA ───────────→ DepthwiseConv → FFN
Hybrid (đề xuất):      FFN → Bi-Mamba ────────→ DepthwiseConv → FFN
                                    └─ B, C được gate bởi F0 embedding
```

**Ý tưởng chính:** tiếng Việt là ngôn ngữ thanh điệu — F0 contour (~60–300 ms)
mang thanh. Mamba quét trạng thái **O(n)** thay vì attention O(n²), vẫn giữ
phụ thuộc dài hạn; DepthwiseConv giữ bias cục bộ (âm vị/phụ âm); F0 gate giúp
phân biệt người nói có cao độ chồng lấn.

> **Đọc bắt đầu từ đâu?** Tài liệu kỹ thuật nằm trong [`docs/`](docs/). Nếu
> bạn muốn *hiểu code khớp với lý thuyết như thế nào*, hãy đọc
> [`docs/code_walkthrough_vi.md`](docs/code_walkthrough_vi.md) trước.

---

## Cấu trúc repository

```
conformer-bimamba/
├── README.md                     ← lối vào duy nhất (file này)
├── docs/                         ← toàn bộ tài liệu (xem mục "Tài liệu" bên dưới)
├── conformer_bimamba/            ← mã nguồn chính (package Python)
│   ├── models/                   ←   ssm / bimamba / hybrid_block / f0_conditioning / separation
│   ├── data/                     ←   manifest, dynamic mixing, dataset
│   ├── preprocess/               ←   trích F0 (PyWorld), sinh mixture
│   └── utils/                    ←   metrics (SI-SNR/PIT), lr, logging, config, seed
├── configs/                      ← 3 cấu hình: bimamba_f0 (đề xuất) / bimamba / conformer
├── train.py  · evaluate.py  · inference.py
├── scripts/download_data.sh
└── train/speech_separation/      ← bản viết lại theo ClearerVoice-Studio (MOSSFormer2-style)
    └── README.md                 ← hướng dẫn riêng của framework đó
```

## Bắt đầu nhanh

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# GPU (Linux+CUDA) để có kernel Mamba nhanh:
#   MAMBA_KEEP_CUDA_BUILD=TRUE pip install causal-conv1d==1.4.0
#   MAMBA_KEEP_CUDA_BUILD=TRUE pip install mamba-ssm==2.0.1 --no-build-isolation

bash scripts/download_data.sh --with_commonvoice   # dữ liệu + F0 + mixtures
python train.py --config configs/train_bimamba_f0.yaml          # huấn luyện
python evaluate.py --config configs/train_bimamba_f0.yaml \
    --checkpoint runs/bimamba-f0/checkpoints/best.pt --perceptual # đánh giá
```

**Huấn luyện theo framework ClearerVoice-Studio / MOSSFormer2** (dùng chung
dataset, checkpoint con trỏ, display): vào [`train/speech_separation/`](train/speech_separation/README.md).

---

## Tài liệu (`docs/`)

| Tài liệu | Nội dung | Ngôn ngữ |
|---|---|---|
| [code_walkthrough_vi.md](docs/code_walkthrough_vi.md) · [**PDF**](docs/code_walkthrough_vi.pdf) | **Hướng dẫn đọc code — ánh xạ từng khối lý thuyết vào từng dòng code** (tái tạo PDF: `python scripts/md2pdf.py docs/code_walkthrough_vi.md docs/code_walkthrough_vi.pdf --title "…"`) | VI |
| [report.md](docs/report.md) | Kiến trúc chi tiết (sơ đồ, công thức SSM, F0 gate) | VI |
| [comparison_vi.md](docs/comparison_vi.md) | So sánh với MOSSFormer/2, SPMamba, SepMamba, DPMamba, TF-GridNet… | VI |
| [report_fsmn_vi.md](docs/report_fsmn_vi.md) | Đề xuất thêm FSMN/GFRN (MOSSFormer2) vào mô hình — ưu/nhược/độ khả quan | VI |
| [parameter_analysis.md](docs/parameter_analysis.md) · [parameter_analysis_vi.md](docs/parameter_analysis_vi.md) | Tham số (đo thật) + dự đoán kết quả | EN · VI |
| [disadvantages.md](docs/disadvantages.md) · [disadvantages_vi.md](docs/disadvantages_vi.md) | Hạn chế & rủi ro trung thực | EN · VI |
| [prediction_vi.md](docs/prediction_vi.md) | Dự đoán thời gian huấn luyện, kết quả theo tham số, so sánh | VI |
| [training_guide_vi.md](docs/training_guide_vi.md) | Hướng dẫn huấn luyện từng bước + xử lý lỗi | VI |
| [colab_training.ipynb](docs/colab_training.ipynb) | Notebook chạy trên Google Colab (quick demo / full, resume qua Drive) | VI |

Ghi chú: code vẫn là nguồn chân lý — các số liệu trong tài liệu (tham số,
MACs, timeline) đều đo từ chính code này; xem cách tái tạo ở cuối
`parameter_analysis.md`.

## Giấy phép & ghi công

- Mã nguồn MOSSFormer2 + framework huấn luyện (trong `train/speech_separation/`):
  © Alibaba / Shengkui Zhao, Apache-2.0 — [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio), [MOSSFormer2](https://arxiv.org/abs/2312.11825).
- PIT/SI-SNR loss: speechbrain. Phần còn lại (model hybrid, F0 conditioning,
  pipeline): dự án này.
