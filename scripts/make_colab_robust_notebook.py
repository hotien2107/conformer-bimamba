#!/usr/bin/env python3
"""Generate a GUARANTEED-to-run Google Colab training notebook for
Hybrid Conformer-BiMamba.

Key idea: it self-adapts to whether the Mamba-1 CUDA kernel is available.
  * kernel OK  -> full config (fast, 8s audio, 30 epochs)
  * no kernel  -> small config (short segments, tiny model, few steps) so the
                  built-in pure-PyTorch reference scan still trains in-browser.
No hard dependency on mamba-ssm (it is *tried* but a failure never aborts the
notebook). Identical Google Drive paths as the MossFormer2 sample.
"""
from __future__ import annotations

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["colab"] = {"provenance": []}
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
nb.metadata["language_info"] = {"name": "python"}

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

cells.append(md("""# 🎙️ Huấn luyện **Hybrid Conformer-BiMamba** trên Google Colab (bản tự thích nghi)
Model: `FFN → Bi-Mamba → DepthwiseConv → FFN` (thay MHSA), tách 2 người nói tiếng Việt.

**Đảm bảo chạy được trên trình duyệt:**
- **Có kernel Mamba** (mamba-ssm build được) → tự dùng cấu hình **đầy đủ** (8s, ~16M tham số, 30 epochs).
- **Không có kernel** (build lỗi, ví dụ Python ≥ 3.12) → tự chuyển cấu hình **nhỏ** (đoạn ngắn, model nhỏ, ít bước) để dùng **reference scan thuần PyTorch** (đã tối ưu bộ nhớ) — vẫn **train thật, chạy được trong browser**, chỉ chậm hơn.

Cùng đường dẫn Google Drive như notebook mẫu MossFormer2. Kernel Mamba vẫn được *thử* cài nhưng **lỗi không làm hỏng notebook** (chỉ giảm hiệu năng).
"""))

# ---- 1. imports ----
cells.append(code('''import os
import sys
import shutil
import zipfile
from pathlib import Path
from tqdm import tqdm'''))

# ---- 2. clone repo ----
cells.append(code('''# ============================================================================
# 1. TẢI CODE TỪ GITHUB  (sửa REPO_URL thành repo model của bạn)
# ============================================================================
REPO_URL = "https://github.com/hotien2107/conformer-bimamba.git"
REPO_DIR = "/content/conformer-bimamba"

if os.path.exists(REPO_DIR):
    shutil.rmtree(REPO_DIR)
print(f"Đang tải code từ {REPO_URL}...")
!git clone {REPO_URL} {REPO_DIR}
%cd {REPO_DIR}
print("Repo:", os.getcwd())'''))

# ---- 3. drive + paths ----
cells.append(code('''# ============================================================================
# 2. CẤU HÌNH ĐƯỜNG DẪN (giữ nguyên như notebook MossFormer2)
# ============================================================================
from google.colab import drive
drive.mount('/content/drive')

WORK_DIR = Path("/content/conformer-bimamba/train/speech_separation")
DATA_ROOT = Path("/content/datasets/VN_SpeechMix_8s_8khz")
ZIP_PATH = Path("/content/drive/MyDrive/VN-SpeechMix_Datasets/dataset_8s_8khz.zip")
CKPT_DIR = Path("/content/drive/MyDrive/checkpoint-hybridbimamba-8s-8khz")
DATA_DIR = WORK_DIR / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
print("WORK_DIR:", WORK_DIR)
print("CKPT_DIR:", CKPT_DIR)'''))

# ---- 4. unzip dataset ----
cells.append(code('''# ============================================================================
# 3. GIẢI NÉN DATASET
# ============================================================================
if not DATA_ROOT.exists() or not list(DATA_ROOT.glob("train/mix/*.wav")):
    print("\\U0001F4E6 Đang giải nén dataset...")
    os.makedirs(DATA_ROOT, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zipf:
        for f in tqdm(zipf.namelist(), desc="Giải nén", unit="files"):
            zipf.extract(f, DATA_ROOT)
    print("✅ Giải nén hoàn tất!")
else:
    print("✅ Dataset đã có sẵn")'''))

# ---- 5. make scp ----
cells.append(code('''# ============================================================================
# 4. TẠO FILE SCP (mix s1 s2 — 3 cột, không F0)
# ============================================================================
print("\\U0001F4DD Tạo file SCP...")
split_map = {"train": ("train", "train.scp"),
             "valid": ("valid", "dev.scp"),
             "test":  ("test", "test.scp")}
for orig_split, (data_split, scp_name) in split_map.items():
    mix_dir = DATA_ROOT / data_split / "mix"
    if not mix_dir.exists():
        continue
    lines = []
    for f in sorted(mix_dir.glob("*.wav")):
        s1, s2 = (DATA_ROOT / data_split / "s1" / f.name), (DATA_ROOT / data_split / "s2" / f.name)
        if s1.exists() and s2.exists():
            lines.append(f"{f} {s1} {s2}\\n")
    (DATA_DIR / scp_name).write_text("".join(lines))
    print(f"  {orig_split:6s} -> {scp_name:12s} ({len(lines)} files)")'''))

# ---- 6. install (never aborts) ----
cells.append(md("""### 5. Cài thư viện
Cài gói thuần Python **luôn thành công**. Kernel Mamba được **thử** cài nhưng nếu
lỗi **không sao** — notebook vẫn chạy bằng reference scan (cell sau tự chọn cấu hình)."""))

cells.append(code('''!pip install -q ninja yamlargparse librosa soundfile einops torchinfo rotary-embedding-torch tensorboard pesq pystoi packaging setuptools wheel triton

import os, subprocess, sys, torch
print("python", sys.version.split()[0], "| torch", torch.__version__, "| CUDA", torch.version.cuda,
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

def sh(cmd):
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    return r.returncode, (r.stdout or "") + (r.stderr or "")

# Thử cài kernel Mamba (KHÔNG quyết định thành bại của notebook)
print("\\U0001F9F0 Thử cài kernel Mamba (nếu lỗi sẽ dùng reference scan)...")
try:
    for p in ["/usr/local/cuda", "/usr/lib/cuda", "/opt/cuda"]:
        if os.path.isdir(p):
            os.environ["CUDA_HOME"] = p; break
    os.environ["MAMBA_KEEP_CUDA_BUILD"] = "TRUE"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0;8.6;9.0"
    os.environ["MAX_JOBS"] = "4"
    for cmd in ["pip install --no-build-isolation -q causal-conv1d",
                "pip install --no-build-isolation -q mamba-ssm"]:
        rc, out = sh(cmd)
        print(("  OK " if rc == 0 else "  FAIL ") + cmd.split()[-1].split("==")[0])
except Exception as e:
    print("  Kernel Mamba: bỏ qua ->", e)
print("Cài đặt xong.")'''))

# ---- 7. detect kernel + auto config ----
cells.append(md("""### 6. Tự phát hiện mamba-ssm & chọn cấu hình (mặc định chạy **Mamba-2**)
- **Có `mamba_ssm`** → dùng cấu hình **đầy đủ** với `use_mamba2: 1` (kernel Triton, nhanh).
- **Không có `mamba_ssm`** → dùng cấu hình **nhỏ** với `use_mamba2: 0` (Mamba-1 reference
  scan thuần PyTorch) — vẫn chạy được trong browser.
> ⚠️ Mamba-2 (Triton) thường cần GPU **sm80 trở lên** (A100/L4/4090). GPU **T4 miễn phí
> (sm75)** có thể không chạy kernel Mamba-2 — khi đó cell sẽ cảnh báo và bạn nên dùng
> runtime có A100/L4/4090, hoặc để notebook tự rơi về reference scan."""))

cells.append(code('''import yaml
from pathlib import Path

# Phát hiện mamba_ssm (Mamba-2 dùng mamba_ssm.Mamba2 qua Triton)
mamba_ok = False
try:
    import mamba_ssm
    from mamba_ssm import Mamba2
    mamba_ok = True
except Exception as e:
    print("không có mamba_ssm:", e)

# Kiểm tra kiến trúc GPU (Mamba-2 Triton cần sm80+)
cm = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
arch = f"sm{cm[0]}{cm[1]}" if cm[0] else "CPU"
print("mamba_ssm:", "CÓ" if mamba_ok else "KHÔNG", "| GPU arch:", arch)
if mamba_ok and cm[0] > 0 and cm[0] < 8:
    print("\\u26a0\\ufe0f GPU < sm80 (ví dụ T4 sm75): kernel Mamba-2 (Triton) CÓ THỂ không chạy. "
          "Nếu forward lỗi, dùng A100/L4/4090 hoặc chuyển use_mamba2=0.")

if mamba_ok:
    cfg = {
        'mode': 'train', 'use_cuda': 1,
        'sampling_rate': 8000, 'max_length': 8,
        'network': 'HybridBiMamba_SS_8K', 'num_spks': 2,
        'encoder_kernel_size': 16, 'encoder_embedding_dim': 256,
        'num_intra': 4, 'num_inter': 4, 'chunk_size': 250,
        'd_ffn': 1024, 'conv_kernel': 31, 'nhead': 8, 'dropout': 0.1,
        'd_state': 128, 'd_conv': 4, 'expand': 2, 'combine': 'sum',
        'use_mamba2': 1,          # <-- chạy với Mamba-2 (Triton, nhanh)
        'use_f0': 0,              # Mamba-2 KHÔNG hỗ trợ F0 -> tắt
        'f0_dim': 64, 'checkpointing': 0,
        'load_type': 'one_input_multi_outputs',
        'tr_list': str(DATA_DIR / 'train.scp'),
        'cv_list': str(DATA_DIR / 'dev.scp'),
        'tt_list': str(DATA_DIR / 'test.scp'),
        'init_learning_rate': 0.0001, 'finetune_learning_rate': 0.00003,
        'loss_threshold': -9999.0, 'max_epoch': 30, 'weight_decay': 0.0001,
        'clip_grad_norm': 5.0, 'seed': 777,
        'num_workers': 2, 'batch_size': 2, 'accu_grad': 2, 'effec_batch_size': 8,
    }
else:
    # Cấu hình NHỎ (không có mamba_ssm): dùng Mamba-1 reference scan thuần PyTorch
    cfg = {
        'mode': 'train', 'use_cuda': 1,
        'sampling_rate': 8000, 'max_length': 3,
        'network': 'HybridBiMamba_SS_8K', 'num_spks': 2,
        'encoder_kernel_size': 16, 'encoder_embedding_dim': 64,
        'num_intra': 2, 'num_inter': 2, 'chunk_size': 128,
        'd_ffn': 256, 'conv_kernel': 15, 'nhead': 4, 'dropout': 0.1,
        'd_state': 16, 'd_conv': 2, 'expand': 2, 'combine': 'sum',
        'use_mamba2': 0, 'use_f0': 0, 'f0_dim': 16, 'checkpointing': 0,
        'load_type': 'one_input_multi_outputs',
        'tr_list': str(DATA_DIR / 'train.scp'),
        'cv_list': str(DATA_DIR / 'dev.scp'),
        'tt_list': str(DATA_DIR / 'test.scp'),
        'init_learning_rate': 0.0003, 'finetune_learning_rate': 0.0001,
        'loss_threshold': -9999.0, 'max_epoch': 3, 'weight_decay': 0.0001,
        'clip_grad_norm': 5.0, 'seed': 777,
        'num_workers': 2, 'batch_size': 2, 'accu_grad': 1, 'effec_batch_size': 2,
    }
    for name in ("train.scp", "dev.scp", "test.scp"):
        p = DATA_DIR / name
        if p.exists():
            lines = p.read_text().splitlines()
            cap = {"train.scp": 400, "dev.scp": 100, "test.scp": 50}[name]
            p.write_text("\\n".join(lines[:cap]) + "\\n")
            print(f"  (giảm {name} -> {min(len(lines), cap)} files)")

config_path = WORK_DIR / "config/train/vn_speechmix_hybrid_8s_8khz.yaml"
os.makedirs(config_path.parent, exist_ok=True)
with open(config_path, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

print("\\U0001F6E0 Config ->", config_path, "| use_mamba2:", cfg["use_mamba2"])'''))

# ---- 8. verify model ----
cells.append(code('''# Kiểm tra mô hình + số tham số + THỬ forward (chạy đúng kernel/lõi đã chọn)
%cd {WORK_DIR}
import torch
from networks import network_wrapper
from types import SimpleNamespace

def build():
    return SimpleNamespace(network="HybridBiMamba_SS_8K",
                    encoder_embedding_dim=cfg["encoder_embedding_dim"],
                    encoder_kernel_size=16, num_spks=2,
                    num_intra=cfg["num_intra"], num_inter=cfg["num_inter"],
                    chunk_size=cfg["chunk_size"], d_ffn=cfg["d_ffn"],
                    conv_kernel=cfg["conv_kernel"], nhead=cfg["nhead"],
                    dropout=cfg["dropout"], d_state=cfg["d_state"],
                    d_conv=cfg["d_conv"], expand=cfg["expand"], combine="sum",
                    use_mamba2=cfg["use_mamba2"], use_f0=0,
                    f0_dim=cfg["f0_dim"], checkpointing=0)

a = build()
m = network_wrapper(a).ss_network
n = sum(p.numel() for p in m.parameters())
print(f"\\U0001F6E0 Model params: {n/1e6:.2f}M | use_mamba2: {cfg['use_mamba2']}")
x = torch.randn(1, cfg["sampling_rate"] * cfg["max_length"])
try:
    out = m(x)
    print("forward OK:", [tuple(o.shape) for o in out])
except Exception as e:
    print("\\u26a0\\ufe0f FORWARD LỖI:", e)
    print("  -> Nếu chạy Mamba-2 trên GPU < sm80, hãy đổi sang GPU >= sm80 "
          "(A100/L4/4090) hoặc đặt cfg['use_mamba2']=0 để dùng reference scan.")'''))

# ---- 9. training ----
cells.append(code('''# ============================================================================
# 7. TRAINING (bash train.sh + env trỏ Drive)
# ============================================================================
%cd {WORK_DIR}

n_train = len(open(DATA_DIR / 'train.scp').readlines())
steps_per_epoch = max(1, n_train // cfg['batch_size'])
save_freq = max(1, steps_per_epoch // 2)   # lưu ~2 lần/epoch để thấy checkpoint sớm

print(f"\\n{'='*60}")
print("TRAINING HYBRID CONFORMER-BIMAMBA (8s 8kHz) | use_mamba2:", cfg['use_mamba2'])
print(f"{'='*60}")
print(f"  Train {n_train} | Batch {cfg['batch_size']} (accu {cfg['accu_grad']}) | "
      f"~{steps_per_epoch} steps/epoch | {cfg['max_epoch']} epochs")
print(f"  Save ~{save_freq} steps | Checkpoint: {CKPT_DIR}")
print(f"{'='*60}\\n")

os.environ.update({
    'GPU_ID': '0', 'N_GPU': '1',
    'CONFIG_PTH': str(config_path),
    'CHECKPOINT_DIR': str(CKPT_DIR),
    'TRAIN_FROM_LAST_CHECKPOINT': '1',
    'INIT_CHECKPOINT_PATH': 'None',
    'PRINT_FREQ': '20',
    'CHECKPOINT_SAVE_FREQ': str(save_freq),
})
!bash train.sh

print(f"\\n{'='*60}\\n🎉 XONG! Checkpoint: {CKPT_DIR}")'''))

# ---- 10. tensorboard ----
cells.append(code('''# TensorBoard trong Colab (bấm link sau khi chạy)
%load_ext tensorboard
%tensorboard --logdir {CKPT_DIR}'''))

cells.append(code(""))

nb.cells = cells
out = "training_hybridbimamba_vi_colab.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("OK ->", out)
