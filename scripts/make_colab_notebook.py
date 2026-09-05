#!/usr/bin/env python3
"""Generate `colab_training.ipynb` — a Google-Colab-ready notebook for the
Hybrid Conformer-BiMamba Vietnamese speech separation project.

Run from the repo root:
    python scripts/make_colab_notebook.py
"""
from __future__ import annotations

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["colab"] = {"provenance": []}
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
nb.metadata["language_info"] = {"name": "python"}

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

cells = []

# ---------------------------------------------------------------------------
cells.append(md("""# 🎙️ Huấn luyện Hybrid Conformer-BiMamba trên Google Colab

Tách giọng nói tiếng Việt (2 người nói) với mô hình **Conformer-BiMamba + điều
kiện hóa F0 (thanh điệu)**.

**Chế độ chạy:**
- `MODE = "quick"` — bản demo (< 1 giờ, GPU T4 miễn phí): dữ liệu con, biến thể
  XS, vài epoch.
- `MODE = "full"` — huấn luyện thật (S hoặc M): cần **Colab Pro/Pro+ (A100)** và
  **nhiều phiên** (mỗi phiên ~12 h); checkpoint lưu vào Google Drive và `train.py`
  hỗ trợ `--resume` nên chạy lại là tiếp tục, không mất tiến trình.

**Lưu ý quan trọng:**
- Kernel Mamba (mamba-ssm) chỉ build trên **Linux + CUDA** — Colab đáp ứng, nhưng
  lần đầu cài mất **5–15 phút biên dịch**. Nếu cài lỗi, notebook tự chuyển sang
  *reference scan* (thuần PyTorch — chậm, chỉ hợp quick mode).
- Dữ liệu, F0 và checkpoint được cache trong Google Drive (`DRIVE_ROOT`) để dùng
  lại giữa các phiên.
- Vivos tải tự động (~1,5 GB, công khai). Common Voice vi **bị gated** — cần
  `HF_TOKEN`; bỏ trống nếu chỉ dùng Vivos.
"""))

cells.append(md("""## ⚙️ 1. Cấu hình chạy
Sửa các giá trị trong ô dưới (bấm vào ô → Edit → Run)."""))

cells.append(code('''# ============================ CẤU HÌNH ============================
MODE       = "quick"          # "quick" (demo <1h) | "full" (thật, nhiều phiên)
VARIANT    = "XS"             # XS | S | M   (M cần A100 + nhiều phiên)
CONFIG_FILE = "configs/train_bimamba_f0.yaml"   # hoặc train_bimamba.yaml / train_conformer.yaml

# Repo: đẩy repository này lên GitHub rồi điền URL, HOẶC để trống và upload
# thư mục conformer-bimamba vào /content (File → Upload to session storage).
GITHUB_URL = "https://github.com/<YOUR_USER>/conformer-bimamba.git"

DRIVE_ROOT = "/content/drive/MyDrive/conformer_bimamba_colab"  # cache + checkpoint
HF_TOKEN   = ""               # cho Common Voice (gated); "" = chỉ dùng Vivos

# full mode:
EPOCHS     = 100              # số epoch tối đa (early stop sẽ dừng sớm hơn)
MAX_STEPS  = 0                # 0 = theo EPOCHS; đặt số để giới hạn bước
N_VAL      = 50               # số mixture val (quick: 50, full: 300)
N_TEST     = 100              # số mixture test (full: 500)
# ========================================================================
'''))

cells.append(md("""## 2. Kiểm tra GPU & kết nối Drive"""))

cells.append(code('''import os, sys, subprocess, torch
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
subprocess.run(["nvidia-smi", "-L"])
from google.colab import drive
drive.mount("/content/drive")
os.makedirs(DRIVE_ROOT, exist_ok=True)
print("Drive OK:", DRIVE_ROOT)'''))

cells.append(md("""## 3. Cài đặt thư viện (lần đầu ~10–20 phút vì biên dịch kernel Mamba)"""))

cells.append(code('''# --- thư viện chung ---
!pip install -q ninja pyworld pesq pystoi datasets einops pyyaml soundfile
# --- kernel Mamba: GIỮ kernel Mamba-1 (cần để chèn F0 vào B, C) ---
import os
os.environ["MAMBA_KEEP_CUDA_BUILD"] = "TRUE"   # quan trọng!
!pip install -q causal-conv1d==1.4.0
!pip install -q mamba-ssm==2.0.1 --no-build-isolation
print("Cài đặt xong (nếu mamba-ssm lỗi, notebook vẫn chạy bằng reference scan — chậm hơn nhiều).")'''))

cells.append(md("""## 4. Lấy mã nguồn"""))

cells.append(code('''import subprocess
def sh(cmd):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)

REPO = "/content/conformer-bimamba"
if not os.path.isdir(REPO):
    if GITHUB_URL and "YOUR_USER" not in GITHUB_URL:
        sh(["git", "clone", GITHUB_URL, REPO])
    else:
        raise RuntimeError("Chưa có repo! Điền GITHUB_URL (mục 1) hoặc upload thư mục vào /content")
sys.path.insert(0, REPO)
os.chdir(REPO)
print("Repo tại:", os.getcwd())'''))

cells.append(code('''from conformer_bimamba.models.ssm import CUDA_SCAN_AVAILABLE
print("CUDA scan (kernel Mamba-1):", CUDA_SCAN_AVAILABLE)
if not CUDA_SCAN_AVAILABLE:
    print("⚠️ Dùng reference scan (thuần PyTorch) — chỉ phù hợp quick mode.")
print("Model params (kiểm tra nhanh):")
from conformer_bimamba.models import HybridSepformer
m = HybridSepformer()   # cấu hình mặc định M
print(f"  HybridSepformer mặc định: {sum(p.numel() for p in m.parameters())/1e6:.2f} M tham số")'''))

cells.append(md("""## 5. Dữ liệu: Vivos + F0 + mixture (cache trong Drive)

- Lần đầu: tải Vivos (~1,5 GB) → trích F0 (PyWorld) → tạo mixture val/test.
- Lần sau: mọi thứ đã nằm trong Drive nên bước này **tự bỏ qua**.
- Quick mode giới hạn số câu để chạy nhanh; full mode dùng toàn bộ Vivos."""))

cells.append(code('''import subprocess
def sh(cmd):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)

DATA = f"{DRIVE_ROOT}/data"; F0DIR = f"{DRIVE_ROOT}/f0"; MIX = f"{DRIVE_ROOT}/mixtures"
for d in (DATA, F0DIR, MIX): os.makedirs(d, exist_ok=True)
LIMIT = 400 if MODE == "quick" else 0
WORKERS = min(4, os.cpu_count() or 2)

# 5.1) Vivos + manifest (bỏ qua nếu đã tải)
if not os.path.exists(f"{DATA}/manifests/vivos_train.jsonl"):
    sh(["python", "-m", "conformer_bimamba.data.download", "--out_dir", DATA, "--f0_root", F0DIR])

from conformer_bimamba.data.vivos import load_manifest, save_manifest
tr = load_manifest(f"{DATA}/manifests/vivos_train.jsonl")
if LIMIT: tr = tr[:LIMIT]

# 5.2) F0 offline — chỉ trích những câu chưa có (cache Drive)
todo = [e for e in tr if not (e.get("f0") and os.path.exists(e["f0"]))]
if todo:
    print(f"Trích F0 cho {len(todo)} câu với {WORKERS} workers (quick ~3–5 phút, đủ 11.660 câu ~20–40 phút)...")
    from conformer_bimamba.preprocess.extract_f0 import extract_manifest
    done = extract_manifest(todo, F0DIR, workers=WORKERS)
    by_path = {e["path"]: e for e in done}
    tr = [by_path.get(e["path"], e) for e in tr]
save_manifest(tr, f"{DATA}/manifests/train_f0.jsonl")
print("Manifest huấn luyện:", len(tr), "câu ->", f"{DATA}/manifests/train_f0.jsonl")

# 5.3) mixture val/test cố định (F0 trích từ mixture thật)
if not os.path.exists(f"{MIX}/val/manifest.json"):
    sh(["python", "-m", "conformer_bimamba.preprocess.make_mixtures",
        "--manifest", f"{DATA}/manifests/vivos_test.jsonl", "--out_dir", f"{MIX}/val",
        "--n_pairs", str(N_VAL), "--sr", "32000", "--dur_sec", "5.0", "--snr_db", "0",
        "--seed", "0", "--workers", str(WORKERS)])
if MODE == "full" and not os.path.exists(f"{MIX}/test/manifest.json"):
    sh(["python", "-m", "conformer_bimamba.preprocess.make_mixtures",
        "--manifest", f"{DATA}/manifests/vivos_test.jsonl", "--out_dir", f"{MIX}/test",
        "--n_pairs", str(N_TEST), "--sr", "32000", "--dur_sec", "5.0", "--snr_db", "0",
        "--seed", "1", "--workers", str(WORKERS)])
print("Mixtures OK:", MIX)'''))

cells.append(md("""## 6. Huấn luyện

- Quick: ~vài phút trên T4. Full: chạy nhiều phiên — **chạy lại ô này là tự
  `--resume` từ checkpoint cuối trong Drive**, nên cứ để nó chạy tới khi ngắt
  kết nối rồi chạy lại ở phiên sau.
- Chỉnh `CONFIG_FILE` ở mục 1 để chạy baseline (`train_conformer.yaml`) hay
  ablation không-F0 (`train_bimamba.yaml`)."""))

cells.append(code('''RUN_NAME = f"cb-{VARIANT}-{MODE}"
OV = [
    f"data.train_manifest={DATA}/manifests/train_f0.jsonl",
    f"data.val_mixtures={MIX}/val",
    f"data.test_mixtures={MIX}/val",          # quick: dùng val làm test
    f"logging.log_dir={DRIVE_ROOT}/runs",
    "logging.use_wandb=false",
    "logging.use_tensorboard=true",
    f"logging.run_name={RUN_NAME}",
]
if VARIANT == "XS":
    OV += ["model.encoder_dim=128", "model.d_ffn=512", "model.num_intra=2", "model.num_inter=2",
           "model.d_state=32", "model.f0_dim=32", "model.expand=1"]
elif VARIANT == "S":
    OV += ["model.encoder_dim=192", "model.d_ffn=768", "model.num_intra=3", "model.num_inter=3",
           "model.d_state=48", "model.f0_dim=48"]
if MODE == "quick":
    OV += ["data.train_dur_sec=2.0", "data.num_workers=2", "train.batch_size=2", "train.grad_accum=1",
           "train.epochs=3", "train.max_steps=60", "train.warmup_steps=5",
           "train.eval_every_epochs=1", "train.save_every_epochs=1",
           "train.early_stop_patience=100"]
else:
    OV += [f"train.epochs={EPOCHS}", f"train.max_steps={MAX_STEPS or 200000}"]

RESUME = f"{DRIVE_ROOT}/runs/{RUN_NAME}/checkpoints/last.pt"
cmd = ["python", "train.py", "--config", CONFIG_FILE] + [f"--override {o}" for o in OV]
if os.path.exists(RESUME):
    cmd += ["--resume", RESUME]
    print("Tiếp tục từ checkpoint:", RESUME)
print("Lệnh:", " ".join(cmd[:4]), "...")
!{" ".join(cmd)}'''))

cells.append(code('''# TensorBoard trong Colab (bấm link sau khi chạy)
%load_ext tensorboard
%tensorboard --logdir {DRIVE_ROOT}/runs'''))

cells.append(md("""## 7. Đánh giá (SI-SNRi, SDRi, PESQ, STOI, latency)"""))

cells.append(code('''BEST = f"{DRIVE_ROOT}/runs/{RUN_NAME}/checkpoints/best.pt"
CKPT = BEST if os.path.exists(BEST) else RESUME
cmd = ["python", "evaluate.py", "--config", CONFIG_FILE] + [f"--override {o}" for o in OV]
cmd += ["--checkpoint", CKPT, "--mixtures", f"{MIX}/val", "--perceptual", "--latency-repeats", "10"]
print("Đánh giá checkpoint:", CKPT)
!{" ".join(cmd)}'''))

cells.append(md("""## 8. Demo suy luận: nghe kết quả tách"""))

cells.append(code('''import glob, subprocess, IPython.display as ipd
mix = sorted(glob.glob(f"{MIX}/val/*/mixture.wav"))[0]
cmd = ["python", "inference.py", "--config", CONFIG_FILE] + [f"--override {o}" for o in OV] + \
      ["--checkpoint", CKPT, "--input", mix, "--out_dir", "separated", "--device", "cuda"]
subprocess.run(cmd, check=True)
print("--- Mixture (đầu vào) ---"); ipd.display(ipd.Audio(mix))
print("--- Ước lượng người nói 1 ---"); ipd.display(ipd.Audio("separated/mixture_s1.wav"))
print("--- Ước lượng người nói 2 ---"); ipd.display(ipd.Audio("separated/mixture_s2.wav"))'''))

cells.append(md("""## 9. Chiến lược nhiều phiên (full mode)

| Việc | Phiên 1 | Phiên 2+ |
|---|---|---|
| Cài kernel Mamba (biên dịch) | 1 lần (10–20 phút) | bỏ qua (đã cài trong máy ảo? **không** — máy ảo Colab reset) |
| Dữ liệu + F0 | cache vào Drive, ~30–60 phút | tự bỏ qua |
| Huấn luyện | chạy tới khi ngắt | chạy lại ô 6 → `--resume` |

⚠️ Máy ảo Colab bị reset mỗi phiên: **mọi thứ ngoài Drive đều mất** (kể cả
gói đã pip cài). Vì vậy phiên sau phải chạy lại ô 3 (cài đặt) — mất ~10–20
phút biên dịch mamba-ssm. Nếu muốn bỏ qua, dùng `MODE=quick` (reference scan,
không cần mamba-ssm).

**Dự kiến thời gian (xem `prediction_vi.md`):** XS ≈ 7–9 h (3090) / ~15 h (T4);
S ≈ 13–17 h (3090) / ~1,5 ngày (T4); **M ≈ 1,5–3,5 ngày (3090) / ~2 ngày
(A100 Pro)** — M chỉ nên chạy trên A100 với nhiều phiên + resume.
"""))

nb.cells = cells

with open("colab_training.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("OK -> colab_training.ipynb")
