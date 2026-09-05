# Hướng dẫn huấn luyện: Hybrid Conformer-BiMamba (tiếng Việt)

*Quy trình từng bước từ cài đặt → dữ liệu → huấn luyện → đánh giá → suy luận,
kèm thời gian dự kiến, cách theo dõi và xử lý lỗi. Thời gian dựa trên
`prediction_vi.md` (RTX 3090 / A10G 24 GB).*

---

## 1. Yêu cầu hệ thống

| Thành phần | Yêu cầu | Ghi chú |
|---|---|---|
| GPU | 1× NVIDIA 24 GB (RTX 3090 / A10G) | A100 40 GB nhanh hơn ~2× |
| Hệ điều hành | **Linux + CUDA 11.8+** | bắt buộc cho kernel Mamba (mamba-ssm) |
| Python | 3.9 – 3.11 | |
| PyTorch | 2.1+ | kernel Mamba cần bộ công cụ C++/CUDA/ninja |
| macOS/Windows | chỉ dùng được **phép quét tham chiếu** (chậm) | phát triển/kiểm thử, không huấn luyện thật |

## 2. Cài đặt

```bash
# 1) môi trường
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# 2) thư viện chung
pip install -r requirements.txt          # torch, torchaudio, pyworld, pesq, pystoi, wandb...

# 3) kernel Mamba (Linux + CUDA; bắt buộc giữ kernel Mamba-1 để chèn F0 vào B, C)
export MAMBA_KEEP_CUDA_BUILD=TRUE        # quan trọng: giữ lại selective_scan_cuda (Mamba-1)
pip install causal-conv1d==1.4.0
pip install mamba-ssm==2.0.1 --no-build-isolation

# kiểm tra kernel đã sẵn sàng
python -c "from conformer_bimamba.models.ssm import CUDA_SCAN_AVAILABLE; print('CUDA scan:', CUDA_SCAN_AVAILABLE)"
# → phải in CUDA scan: True
```

> Nếu build lỗi: cài lại với `MAMBA_KEEP_CUDA_BUILD=TRUE pip install mamba-ssm --no-build-isolation`,
> kiểm tra `nvcc --version` và `torch.version.cuda` khớp nhau.

## 3. Chuẩn bị dữ liệu (khoảng 1–2 giờ)

```bash
# tải Vivos (+ tùy chọn Common Voice vi), dựng manifest, trích F0, tạo mixture val/test
bash scripts/download_data.sh --with_commonvoice
```

Chi tiết từng bước (hoặc chạy thủ công):

```bash
# 3.1) tải corpus + manifest (Vivos công khai; Common Voice cần HF_TOKEN — gated)
python -m conformer_bimamba.data.download --out_dir data --f0_root data/f0

# 3.2) trích F0 offline (PyWorld, song song; ~15–40 phút cho 11.660 câu với 8 workers)
python -m conformer_bimamba.preprocess.extract_f0 \
    --manifest data/manifests/vivos_train.jsonl --out_dir data/f0 \
    --out_manifest data/manifests/vivos_train_f0.jsonl --workers 8
cp data/manifests/vivos_train_f0.jsonl data/manifests/train_f0.jsonl   # manifest huấn luyện

# 3.3) mixture cố định cho val/test (F0 trích từ mixture thật — đúng những gì mô hình thấy lúc suy luận)
python -m conformer_bimamba.preprocess.make_mixtures \
    --manifest data/manifests/vivos_test.jsonl --out_dir mixtures/val \
    --n_pairs 300 --sr 32000 --dur_sec 5.0 --snr_db 0 --seed 0 --workers 8
python -m conformer_bimamba.preprocess.make_mixtures \
    --manifest data/manifests/vivos_test.jsonl --out_dir mixtures/test \
    --n_pairs 500 --sr 32000 --dur_sec 5.0 --snr_db 0 --seed 1 --workers 8
```

Cấu trúc dữ liệu tạo ra:

```
data/
├── manifests/          # jsonl: {speaker, path, duration, f0}
│   └── train_f0.jsonl  # → trỏ vào configs/*.yaml (data.train_manifest)
├── f0/                 # data/f0/*.npy — F0 Hz, khung 10 ms, 0 = vô thanh
└── vivos/ test/…       # audio gốc 16 kHz
mixtures/val, mixtures/test/   # mỗi cặp: mixture.wav + s1.wav + s2.wav + f0.npy + meta.json
```

## 4. Cấu hình huấn luyện

Ba cấu hình mặc định trong `configs/`:

| File | Mô hình | Mục đích |
|---|---|---|
| `train_bimamba_f0.yaml` | **Hybrid + F0 (đề xuất)** | chạy chính |
| `train_bimamba.yaml` | Hybrid không F0 (Mamba-2) | ablation 2 |
| `train_conformer.yaml` | Vanilla Conformer (MHSA) | baseline ablation 1 |

Sửa `data.train_manifest` (hoặc dùng `--override`):

```bash
python train.py --config configs/train_bimamba_f0.yaml \
    --override data.train_manifest=data/manifests/train_f0.jsonl
```

Tham số quan trọng (`train.*` / `model.*`): `lr=0.0015`, `mamba_lr_scale=0.5`
(ổn định SSM), `batch_size=4`, `grad_accum=2`, `warmup_steps=5000`,
`amp=true`, `early_stop_patience=10`, `eval_every_epochs=2`.

## 5. Huấn luyện

```bash
# baseline trước (ngày 1) → ablation 2 (ngày 2) → đề xuất F0 (ngày 2–4)
python train.py --config configs/train_conformer.yaml --override logging.run_name=conformer-baseline
python train.py --config configs/train_bimamba.yaml    --override logging.run_name=bimamba-nof0
python train.py --config configs/train_bimamba_f0.yaml --override logging.run_name=bimamba-f0

# tiếp tục từ checkpoint (tắt máy/gián đoạn không mất tiến trình)
python train.py --config configs/train_bimamba_f0.yaml \
    --resume runs/bimamba-f0/checkpoints/last.pt
```

**Theo dõi trong lúc chạy:**
- `[train] params: 16.43M` — đúng nếu kernel + cấu hình đúng.
- Loss PIT SI-SNR giảm dần; ban đầu ~0 dB rồi xuống âm (âm hơn = tốt hơn);
  dừng sớm nếu val không cải thiện sau 10 lần đánh giá.
- Checkpoint: `runs/<run_name>/checkpoints/{best,last,epoch_N}.pt` (model +
  optimizer + scheduler + config).
- TensorBoard: `tensorboard --logdir runs` · WandB: xem dashboard theo project
  `conformer-bimamba`.

**Thời gian dự kiến (3090, xem `prediction_vi.md`):** XS ~7–9 h, S ~13–17 h,
M ~1,5–3,5 ngày, L không khả thi trên 24 GB.

## 6. Đánh giá

```bash
python evaluate.py --config configs/train_bimamba_f0.yaml \
    --checkpoint runs/bimamba-f0/checkpoints/best.pt \
    --mixtures mixtures/test --perceptual --latency-repeats 20 --cpu-latency \
    --out-json results/bimamba_f0.json
```

Đầu ra: SI-SNRi / SDRi (dB), PESQ, STOI, latency GPU/CPU (ms), peak memory (MiB).
Chạy cùng lệnh với 3 config để lập bảng ablation; so sánh cặp đôi trên từng
mixture (không chỉ trung bình) vì nhiễu ±0,5 dB.

## 7. Suy luận (inference)

```bash
# một file hoặc cả thư mục; F0 được trích tự động bằng PyWorld
python inference.py --config configs/train_bimamba_f0.yaml \
    --checkpoint runs/bimamba-f0/checkpoints/best.pt \
    --input audio/mixture.wav --out_dir separated --device cuda
# → separated/mixture_s1.wav, mixture_s2.wav, mixture_mixture.wav
```

## 8. Theo dõi thí nghiệm (Phase 5)

- `configs/*.yaml` → `logging.use_wandb` / `use_tensorboard`; mỗi checkpoint lưu
  kèm toàn bộ config → tái lập kết quả chính xác.
- Seed cố định `seed: 42` (Python/NumPy/torch/cuDNN).
- Val/test dùng **mixture cố định** (sinh 1 lần với seed) → so sánh công bằng
  giữa các lần chạy.

## 9. Xử lý sự cố

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `CUDA scan: False` | mamba-ssm chưa build kernel Mamba-1 | cài lại với `MAMBA_KEEP_CUDA_BUILD=TRUE`; kiểm tra torch/CUDA khớp |
| `RuntimeError: split_with_sizes...` | cấu hình sai d_state/expand | dùng đúng config mặc định |
| Loss = NaN giữa chừng | underflow `dt` fp16 / LR quá cao | giảm `train.lr`, tăng `warmup_steps`; chạy fp32 (`--override train.amp=false`) để chẩn đoán |
| OOM 24 GB | batch quá lớn hoặc L variant | giảm `batch_size`/`train_dur_sec`; bật `model.checkpointing=true` |
| SI-SNRi không cải thiện sau 5 epoch | mamba_lr_scale quá thấp / dữ liệu lỗi | kiểm tra manifest, tăng `mamba_lr_scale` lên 0.75 |
| F0 mất hết (ablation 2 ≈ 3) | F0 gate sụp đổ (‖gate‖→0) | log chuẩn `f0_gate.weight`; kiểm tra dữ liệu F0 (nhiều khung = 0?) |
| pyworld báo lỗi | thiếu `pyworld` / âm thanh rỗng | `pip install pyworld`; lọc câu rỗng khỏi manifest |

## 10. Lộ trình nhanh (24 GB, 3–5 ngày)

1. **Ngày 1:** cài đặt + dữ liệu + F0 + mixtures (mục 2–3); chạy thử 50 bước
   để đo `time/step` thật (điều chỉnh `max_steps` cho khớp dự đoán).
2. **Ngày 1–2:** baseline Conformer-M (19–39 h) — nền để so sánh.
3. **Ngày 2–3:** Hybrid-noF0-M (1,5–3,5 ngày, chạy song song nếu đủ VRAM).
4. **Ngày 2–5:** Hybrid+F0-M — mô hình chính; resume nếu gián đoạn.
5. **Ngày 5:** đánh giá 3 mô hình trên cùng test set, viết báo cáo.

Xem thêm: [`report.md`](report.md) (kiến trúc), [`comparison_vi.md`](comparison_vi.md)
(so sánh mô hình), [`prediction_vi.md`](prediction_vi.md) (dự đoán thời gian/kết quả),
[`parameter_analysis_vi.md`](parameter_analysis_vi.md) (tham số), [`disadvantages_vi.md`](disadvantages_vi.md) (hạn chế),
[`colab_training.ipynb`](colab_training.ipynb) (chạy trên Google Colab).
