# Hybrid Conformer-BiMamba — ClearerVoice-Studio style training

Model and training pipeline rewritten to follow **exactly the same
structure, training method, checkpoint storage and display** as
[`modelscope/ClearerVoice-Studio`](https://github.com/modelscope/ClearerVoice-Studio/tree/main/train/speech_separation)
(the framework that hosts MOSSFormer2's training code).

| network name | model | sampling rate | F0 (tone) conditioning |
|---|---|---|---|
| `MossFormer2_SS_16K` / `MossFormer2_SS_8K` | MOSSFormer2 (original, vendored Apache-2.0) | 16 k / 8 k | no |
| `HybridBiMamba_SS_16K` / `HybridBiMamba_SS_8K` | **Hybrid Conformer-BiMamba** (this project) | 16 k / 8 k | `use_f0: 1` (Vietnamese) |

The HybridBiMamba model replaces MOSSFormer2's joint local-global attention
+ FSMN sequence modelling with a **dual-path stack of Hybrid blocks**
(`FFN → Bi-Mamba → DepthwiseConv → FFN`, i.e. MHSA replaced by bidirectional
Mamba), and optionally conditions the Mamba B/C projections on the **mixture
F0 contour** — designed for the tonal Vietnamese language.

## Directory layout (identical to ClearerVoice-Studio)

```
train/speech_separation/
├── config/{train,inference}/*.yaml     # per-network configs
├── data/                               # .scp generation tool (prepare_vi_mixtures.py)
├── dataloader/{dataloader,misc}.py     # + optional 4th F0 column in .scp
├── losses/loss.py                      # PIT SI-SNR (speechbrain-style)
├── models/
│   ├── mossformer2/                    # vendored original baseline (Apache-2.0)
│   └── hybrid_bimamba/                 # this project's model (ssm/bimamba/hybrid_block/...)
├── networks.py                         # network registry (network_wrapper)
├── solver.py                           # training loop + checkpoints + TensorBoard
├── train.py / train.sh                 # entry points
├── inference.py / inference.sh         # decoding entry points
└── utils/                              # checkpoint reload, segmental decode, metrics
```

## Conventions kept identical to ClearerVoice-Studio

* **Checkpoint storage**: every save writes `model.ckpt-{epoch}-{step}.pt`
  with keys `{model, optimizer, epoch, step}`, then updates a **pointer file**
  (`last_checkpoint` / `last_best_checkpoint`) containing the model file name;
  resuming reads the pointer file (`--train_from_last_checkpoint 1`).
* **Training method**: per-epoch train/val(/test) passes; per-utterance loss
  filtering via `loss_threshold`; gradient accumulation
  (`accu_grad`, `effec_batch_size`); grad-norm clip 10; validation-based
  **learning-rate halving every 5 non-improving epochs** (reloading the best
  checkpoint) and **early stop at 10**; Adam with `init_learning_rate`
  (1.5e-4) / `finetune_learning_rate` (5e-5); distributed (DDP) ready.
* **Display**: identical console prints
  (`Train Summary | End of Epoch x | Time | Train Loss`, per-step
  `Train Epoch: x/y Step: a/b | s/batch | lr | Total_Loss`) and TensorBoard
  scalars `Train_loss / Validation_loss / Test_loss` under
  `<checkpoint_dir>/tensorboard/`.

## Install

```bash
pip install torch torchaudio yamlargparse librosa soundfile pyworld pesq pystoi \
            numpy einops tensorboard rotary-embedding-torch torchinfo
# GPU (Linux + CUDA) — needed for real training speed:
export MAMBA_KEEP_CUDA_BUILD=TRUE
pip install causal-conv1d==1.4.0 mamba-ssm==2.0.1 --no-build-isolation
```

## Data

```bash
# 1) Vivos (+ optional Common Voice vi) manifests and source-F0: use the
#    project-level pipeline (../../data, ../../preprocess) or your own, then:
python data/prepare_vi_mixtures.py \
    --manifest_train /path/to/vivos_train.jsonl \
    --manifest_test  /path/to/vivos_test.jsonl \
    --out_dir data --n_train 20000 --n_cv 500 --n_tt 1000 --workers 8
# -> data/tr_vi_2mix_16k.scp, cv_..., tt_...  (4 columns: mix s1 s2 f0)
```

Each `.scp` line is `mixture.wav s1.wav s2.wav mixture_f0.npy`; the F0
contour (Hz @ 100 fps, 0 = unvoiced) is used only when `use_f0: 1`.

> Using another corpus (e.g. WSJ0-2Mix)? Just point `tr_list`/`cv_list` at
> 3-column `.scp` files and set `use_f0: 0`.

> ⚠️ CLI arg spelling follows the original ClearerVoice-Studio convention
> exactly (mixed hyphens/underscores, e.g. `--tr-list`, `--cv-list`,
> `--use-cuda`, `--max-epoch`, but `--batch_size`, `--num_workers`,
> `--max_length`, `--print_freq`, `--checkpoint_save_freq`,
> `--train_from_last_checkpoint`, `--init_learning_rate`). The yaml config is
> the canonical place to set values; when overriding on the CLI, copy the
> spelling shown by `python train.py --help`.

## Train

```bash
cd train/speech_separation
./train.sh                                   # HybridBiMamba_SS_16K (use_f0: 1)
# or
python train.py --config config/train/HybridBiMamba_SS_16K.yaml \
    --checkpoint_dir checkpoints/HybridBiMamba_SS_16K
# resume:
python train.py --config config/train/HybridBiMamba_SS_16K.yaml \
    --checkpoint_dir checkpoints/HybridBiMamba_SS_16K --train_from_last_checkpoint 1
# original baseline (same framework):
./train_mossformer2.sh   # or: python train.py --config config/train/MossFormer2_SS_16K.yaml
```

To train the **ablation without F0**: `python train.py --config ... --use_f0 0`
(or comment the 4th scp column and keep `use_f0: 0`).

## Evaluate & display results

```bash
./inference.sh                    # separates data/tt_vi_2mix_16k.scp -> outputs/
python utils/eval_objective.py --wav_list <utt list> \
    --pathc <clean dir> --pathn <mixture dir> --pathe <separated dir> \
    --result_list results.csv     # PESQ / STOI / SI-SDR per utterance
python utils/get_results.py results.csv   # averaged table
tensorboard --logdir checkpoints/HybridBiMamba_SS_16K/tensorboard
```

## Attribution

* MOSSFormer2 model + training framework: © Alibaba / Shengkui Zhao, Apache-2.0
  ([ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio),
  [MOSSFormer2 paper](https://arxiv.org/abs/2312.11825)).
* Loss/PIT wrapper: speechbrain.
* Hybrid Conformer-BiMamba + F0 conditioning: this project
  (see `../../README.md` and `../../report.md`).
