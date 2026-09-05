#!/usr/bin/env python3
"""Generate `training_hybridbimamba_vimd_8s_8khz.ipynb`.

A Colab training notebook for the Hybrid Conformer-BiMamba model, written to
mirror `training_mossformer2_vimd_8s_8khz.ipynb` (same repo/drive layout, same
VN_SpeechMix 8s 8kHz dataset, same Google Drive checkpoint path) but driving
the HybridBiMamba_SS_8K network instead of MossFormer2_SS_8K.
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

cells.append(md("""# 🎙️ Huấn luyện **Hybrid Conformer-BiMamba** (8 s, 8 kHz)
Đây là notebook cho **một mô hình khác (Conformer-BiMamba)**, chỉ dùng lại **cùng
pipeline & cùng đường dẫn Drive** với notebook mẫu
[`training_mossformer2_vimd_8s_8khz.ipynb`](training_mossformer2_vimd_8s_8khz.ipynb)
(dataset `VN_SpeechMix_8s_8khz`, thư mục checkpoint trên Drive).
**Model KHÔNG liên quan đến MOSSFormer** — kiến trúc là `FFN → Bi-Mamba →
DepthwiseConv → FFN` (thay MHSA), không có attention/FSMN của MOSSFormer.

> **Chú ý:** lần đầu mamba-ssm sẽ **biên dịch ~10–20 phút** trên GPU Colab; nếu
> cài lỗi, notebook vẫn chạy được bằng *reference scan* (chậm, chỉ để test).
"""))

# ---- 0. imports ----
cells.append(code('''import os
import sys
import shutil
import zipfile
from pathlib import Path
from tqdm import tqdm'''))

# ---- 1. clone repo ----
cells.append(code('''# ============================================================================
# 1. TẢI CODE TỪ GITHUB
# ============================================================================
# 🔧 Sửa URL dưới đây thành repo chứa model của bạn (đã up lên GitHub).
REPO_URL = "https://github.com/hotien2107/conformer-bimamba.git"
REPO_DIR = "/content/conformer-bimamba"

if os.path.exists(REPO_DIR):
    print("Đang xóa thư mục cũ...")
    shutil.rmtree(REPO_DIR)

print(f"Đang tải code từ {REPO_URL}...")
!git clone {REPO_URL} {REPO_DIR}

%cd {REPO_DIR}'''))

# ---- 2. drive mount + paths ----
cells.append(code('''# ============================================================================
# 2. CẤU HÌNH ĐƯỜNG DẪN  (giữ nguyên như notebook mẫu MossFormer2)
# ============================================================================
from google.colab import drive
drive.mount('/content/drive')

# Thư mục chứa framework huấn luyện (ClearerVoice-Studio style)
WORK_DIR = Path("/content/conformer-bimamba/train/speech_separation")

# Dataset VN-SpeechMix 8s 8kHz (trộn sẵn 2 người nói)
DATA_ROOT = Path("/content/datasets/VN_SpeechMix_8s_8khz")
ZIP_PATH = Path("/content/drive/MyDrive/VN-SpeechMix_Datasets/dataset_8s_8khz.zip")

# Checkpoint lưu vào Drive (CÙNG đường dẫn với notebook mẫu; đây chỉ là TÊN
# thư mục lưu, KHÔNG liên quan đến kiến trúc MOSSFormer)
CKPT_DIR = Path("/content/drive/MyDrive/checkpoint-mossformer-lite-8s-8khz")

DATA_DIR = WORK_DIR / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
print("WORK_DIR  :", WORK_DIR)
print("DATA_DIR  :", DATA_DIR)
print("CKPT_DIR  :", CKPT_DIR)'''))

# ---- 3. unzip dataset ----
cells.append(code('''# ============================================================================
# 3. GIẢI NÉN DATASET
# ============================================================================
if not DATA_ROOT.exists() or not list(DATA_ROOT.glob("train/mix/*.wav")):
    print(f"\\U0001F4E6 Đang giải nén dataset...")
    os.makedirs(DATA_ROOT, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zipf:
        for f in tqdm(zipf.namelist(), desc="Giải nén", unit="files"):
            zipf.extract(f, DATA_ROOT)
    print("✅ Giải nén hoàn tất!")
else:
    print("✅ Dataset đã có sẵn")'''))

# ---- 4. make scp ----
cells.append(code('''# ============================================================================
# 4. TẠO FILE SCP  (mix s1 s2  — 3 cột, không F0 -> train model với use_f0=0)
# ============================================================================
print(f"\\U0001F4DD Tạo file SCP...")
split_map = {
    "train": ("train", "train.scp"),
    "valid": ("valid", "dev.scp"),
    "test":  ("test", "test.scp"),
}
for orig_split, (data_split, scp_name) in split_map.items():
    mix_dir = DATA_ROOT / data_split / "mix"
    s1_dir = DATA_ROOT / data_split / "s1"
    s2_dir = DATA_ROOT / data_split / "s2"
    if not mix_dir.exists():
        continue
    lines = []
    for f in sorted(mix_dir.glob("*.wav")):
        s1, s2 = s1_dir / f.name, s2_dir / f.name
        if s1.exists() and s2.exists():
            lines.append(f"{f} {s1} {s2}\\n")
    scp_path = DATA_DIR / scp_name
    scp_path.write_text("".join(lines))
    print(f"  {orig_split:6s} → {scp_name:12s} ({len(lines)} files)")'''))

# ---- 5. install libs (+mamba-ssm) ----
cells.append(code('''# ============================================================================
# 5. CÀI ĐẶT THƯ VIỆN
# ============================================================================
print(f"\\U0001F4E6 Cài đặt thư viện...")
!pip install -q yamlargparse librosa soundfile einops torchinfo rotary-embedding-torch tensorboard pesq pystoi

# Kernel Mamba (chỉ Linux + CUDA). Giữ kernel Mamba-1 (cần cho F0 injection).
import os
os.environ["MAMBA_KEEP_CUDA_BUILD"] = "TRUE"
print("Cài kernel Mamba (biên dịch ~10-20 phút)...")
!pip install -q causal-conv1d==1.4.0
!pip install -q "mamba-ssm==2.0.1" --no-build-isolation
print("Xong. Nếu bước này LỖI, notebook vẫn chạy bằng reference scan (chậm).")'''))

# ---- 6. config ----
cells.append(code('''# ============================================================================
# 6. CONFIG  (HybridBiMamba_SS_8K)
# ============================================================================
import yaml

config = {
    # === Chế độ ===
    'mode': 'train',
    'use_cuda': 1,

    # === Audio (giống mẫu: 8 kHz, 8 giây) ===
    'sampling_rate': 8000,
    'max_length': 8,

    # === Model Architecture (Hybrid Conformer-BiMamba — KHÔNG dùng MossFormer) ===
    'network': 'HybridBiMamba_SS_8K',
    'num_spks': 2,

    # Encoder
    'encoder_kernel_size': 16,
    'encoder_embedding_dim': 256,

    # Dual-path: FFN → Bi-Mamba → DepthwiseConv → FFN (mọi tham số mô hình ở đây)
    'num_intra': 8,                     # số khối hybrid đường trong-đoạn
    'num_inter': 8,                     # số khối hybrid đường liên-đoạn
    'chunk_size': 250,
    'd_ffn': 1024,
    'conv_kernel': 31,
    'nhead': 8,
    'dropout': 0.1,
    'd_state': 64,
    'd_conv': 4,
    'expand': 2,
    'combine': 'sum',
    'use_mamba2': 0,                    # kernel Mamba-1 (hỗ trợ F0 về sau)
    'use_f0': 0,                        # dataset không có cột F0 -> 0
    'f0_dim': 64,
    'checkpointing': 0,

    # === Data Loading ===
    'load_type': 'one_input_multi_outputs',
    'tr_list': str(DATA_DIR / 'train.scp'),
    'cv_list': str(DATA_DIR / 'dev.scp'),
    'tt_list': str(DATA_DIR / 'test.scp'),

    # === Training ===
    'init_learning_rate': 0.0001,
    'finetune_learning_rate': 0.00003,
    'loss_threshold': -9999.0,
    'max_epoch': 30,
    'weight_decay': 0.0001,
    'clip_grad_norm': 5.,
    'seed': 777,

    # === DataLoader ===
    'num_workers': 4,
    'batch_size': 4,
    'accu_grad': 2,
    'effec_batch_size': 16,
}

config_path = WORK_DIR / "config/train/vn_speechmix_hybrid_8s_8khz.yaml"
os.makedirs(config_path.parent, exist_ok=True)
with open(config_path, 'w') as f:
    yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)

print(f"\\u2699\\ufe0f Config saved: {config_path}")
print("\\U0001F4C4 Nội dung config:")
print(open(config_path).read())'''))

# ---- 7. training ----
cells.append(code('''# ============================================================================
# 7. TRAINING  (giống mẫu: `bash train.sh` + env trỏ Drive)
# ============================================================================
%cd {WORK_DIR}

n_train = len(open(DATA_DIR / 'train.scp').readlines())
n_valid = len(open(DATA_DIR / 'dev.scp').readlines())
steps_per_epoch = n_train // config['batch_size']
save_freq = steps_per_epoch * 2

print(f"\\n{'='*60}")
print("🚀 TRAINING HYBRID CONFORMER-BIMAMBA (8s 8kHz)")
print(f"{'='*60}")
print("  Dataset   : VN-SpeechMix 8s 8kHz")
print("  Model     : HybridBiMamba_SS_8K (Bi-Mamba thay MHSA)")
print(f"  Train     : {n_train} files | Valid: {n_valid} files")
print(f"  Batch     : {config['batch_size']}  (accu {config['accu_grad']})")
print(f"  Steps/epoch: ~{steps_per_epoch}")
print(f"  Max epochs: {config['max_epoch']}")
print(f"  Save      : mỗi {save_freq} steps (~2 epochs)")
print(f"  Checkpoint: {CKPT_DIR}")
print(f"{'='*60}\\n")

os.environ.update({
    'GPU_ID': '0',
    'N_GPU': '1',
    'CONFIG_PTH': str(config_path),
    'CHECKPOINT_DIR': str(CKPT_DIR),
    'TRAIN_FROM_LAST_CHECKPOINT': '1',     # tự resume từ checkpoint trong Drive
    'INIT_CHECKPOINT_PATH': 'None',
    'PRINT_FREQ': '50',
    'CHECKPOINT_SAVE_FREQ': str(save_freq),
})

!bash train.sh

print(f"\\n{'='*60}")
print("🎉 TRAINING HOÀN THÀNH!")
print(f"  Checkpoints: {CKPT_DIR}")
print("  TensorBoard: tensorboard --logdir " + str(CKPT_DIR))
print(f"{'='*60}")'''))

cells.append(code(""))  # trailing empty cell (matches sample)

nb.cells = cells

with open("training_hybridbimamba_vimd_8s_8khz.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("OK -> training_hybridbimamba_vimd_8s_8khz.ipynb")
