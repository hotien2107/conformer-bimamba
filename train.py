#!/usr/bin/env python3
"""Train the Hybrid Conformer-BiMamba separator.

Usage:
    python train.py --config configs/train_bimamba_f0.yaml
    python train.py --config configs/train_bimamba_f0.yaml --override train.batch_size=6 --override train.lr=1e-3
    python train.py --config configs/train_conformer.yaml --resume runs/bimamba-f0/checkpoints/last.pt

Pipeline: dynamic 2-speaker mixing -> dual-path separator -> PIT SI-SNR loss
-> AdamW (per-group LR: mamba params at ``mamba_lr_scale``) + warmup/cosine
-> AMP (fp16 on CUDA) -> WandB/TensorBoard + periodic fixed-mixture eval.
"""
from __future__ import annotations

import argparse
import copy
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from conformer_bimamba.data import FixedMixtureDataset, SpeechSeparationDataset
from conformer_bimamba.models import HybridSepformer
from conformer_bimamba.models.ssm import CUDA_SCAN_AVAILABLE, F0ConditionedSSM, Mamba2Core
from conformer_bimamba.utils.audio import rms
from conformer_bimamba.utils.config import load_config
from conformer_bimamba.utils.logging import ExperimentLogger
from conformer_bimamba.utils.lr_scheduler import WarmupCosineSchedule
from conformer_bimamba.utils.metrics import si_snr_loss, si_snri, sdri
from conformer_bimamba.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Model / optimizer
# ---------------------------------------------------------------------------
def build_model(cfg: dict) -> HybridSepformer:
    m = cfg["model"]
    return HybridSepformer(
        encoder_dim=m["encoder_dim"],
        encoder_kernel=m["encoder_kernel"],
        encoder_stride=m["encoder_stride"],
        chunk_size=m["chunk_size"],
        hop=m["hop"],
        block_type=m["block_type"],
        num_intra=m["num_intra"],
        num_inter=m["num_inter"],
        d_ffn=m["d_ffn"],
        conv_kernel=m["conv_kernel"],
        nhead=m["nhead"],
        dropout=m["dropout"],
        use_f0=m["use_f0"],
        f0_dim=m["f0_dim"],
        d_state=m["d_state"],
        d_conv=m["d_conv"],
        expand=m["expand"],
        combine=m["combine"],
        use_mamba2=m["use_mamba2"],
        checkpointing=m["checkpointing"],
    )


def build_optimizer(model: HybridSepformer, cfg: dict) -> torch.optim.Optimizer:
    """AdamW with a separate (lower) LR group for SSM parameters (risk R1)."""
    t = cfg["train"]
    ssm_ids = set()
    for module in model.modules():
        if isinstance(module, (F0ConditionedSSM, Mamba2Core)):
            for p in module.parameters():
                ssm_ids.add(id(p))
    mamba_params = [p for p in model.parameters() if id(p) in ssm_ids]
    rest_params = [p for p in model.parameters() if id(p) not in ssm_ids]
    groups = [
        {"params": mamba_params, "lr": t["lr"] * t["mamba_lr_scale"]},
        {"params": rest_params, "lr": t["lr"]},
    ]
    return torch.optim.AdamW(groups, weight_decay=t["weight_decay"])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: HybridSepformer, loader: DataLoader, device: torch.device, amp: bool) -> dict:
    model.eval()
    si_snris, sdris = [], []
    for batch in loader:
        mixture = batch["mixture"].to(device)
        sources = batch["sources"].to(device)
        f0 = batch["f0"].squeeze(-1).to(device) if batch.get("f0") is not None else None
        with torch.autocast("cuda", enabled=amp):
            est = model(mixture, f0)
        T = min(est.shape[-1], sources.shape[-1])
        est = est[..., :T].squeeze(2)
        ref = sources[..., :T]
        si_snris.append(si_snri(est, ref, mixture))
        sdris.append(sdri(est, ref, mixture))
    model.train()
    return {
        "si_snri": torch.cat(si_snris).mean().item(),
        "sdri": torch.cat(sdris).mean().item(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", action="append", default=[], help="dotted.key=value (repeatable)")
    ap.add_argument("--resume", default=None, help="checkpoint path")
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    set_seed(cfg["seed"])
    d, m, t, lg = cfg["data"], cfg["model"], cfg["train"], cfg["logging"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(t["amp"]) and device.type == "cuda"
    print(f"[train] device={device} amp={amp} cuda_scan={CUDA_SCAN_AVAILABLE}")

    # --- data -----------------------------------------------------------
    train_ds = SpeechSeparationDataset(
        entries=_load_manifest_or_build(d, cfg),
        sample_rate=d["sample_rate"],
        dur_sec=d["train_dur_sec"],
        snr_range=tuple(d["snr_range"]),
        f0_mix_mode=d["f0_mix_mode"],
    )
    train_loader = DataLoader(
        train_ds, batch_size=t["batch_size"], shuffle=True,
        num_workers=d["num_workers"], drop_last=True, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        FixedMixtureDataset(d["val_mixtures"]), batch_size=1, shuffle=False, num_workers=2
    )

    # --- model / optim / sched -------------------------------------------
    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    print(f"[train] params: {model.num_parameters()/1e6:.2f}M")

    start_epoch, global_step = 0, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch, global_step = ckpt["epoch"] + 1, ckpt["step"]
        print(f"[train] resumed from {args.resume} (epoch {start_epoch})")

    steps_per_epoch = len(train_loader) // t["grad_accum"]
    total_steps = min(t["max_steps"], t["epochs"] * steps_per_epoch)
    scheduler = WarmupCosineSchedule(optimizer, t["warmup_steps"], total_steps)

    logger = ExperimentLogger(
        project=lg["project"], run_name=lg["run_name"], config=cfg,
        log_dir=lg["log_dir"], use_wandb=lg["use_wandb"], use_tensorboard=lg["use_tensorboard"],
    )
    ckpt_dir = Path(lg["log_dir"]) / lg["run_name"] / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best_si_snri, no_improve = -1e9, 0
    tr = time.time()

    for epoch in range(start_epoch, t["epochs"]):
        model.train()
        ep_loss, ep_steps = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_loader):
            mixture = batch["mixture"].to(device)
            sources = batch["sources"].to(device)
            f0 = batch["f0"].squeeze(-1).to(device) if m["use_f0"] else None

            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                est = model(mixture, f0)
                T = min(est.shape[-1], sources.shape[-1])
                loss = si_snr_loss(est[..., :T].squeeze(2), sources[..., :T])

            loss = loss / t["grad_accum"]
            scaler.scale(loss).backward()

            if (i + 1) % t["grad_accum"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                ep_loss += loss.item() * t["grad_accum"]
                ep_steps += 1
                if global_step % 50 == 0:
                    lr_now = optimizer.param_groups[0]["lr"]
                    logger.log_dict(
                        {"loss": ep_loss / ep_steps, "lr": lr_now,
                         "mixture_rms": rms(mixture).mean().item()},
                        step=global_step, prefix="train",
                    )
                    print(f"[{epoch}] step {global_step}/{total_steps} loss {ep_loss/ep_steps:.3f} "
                          f"lr {lr_now:.2e} ({time.time()-tr:.0f}s)", flush=True)

        # --- validation ---------------------------------------------------
        if (epoch + 1) % t["eval_every_epochs"] == 0:
            val_metrics = evaluate(model, val_loader, device, amp)
            logger.log_dict(val_metrics, step=global_step, prefix="val")
            print(f"[val @ epoch {epoch}] {val_metrics}")

            if (epoch + 1) % t["audio_log_every_epochs"] == 0:
                _log_audio_sample(model, val_loader, logger, device, amp, global_step)

            if val_metrics["si_snri"] > best_si_snri:
                best_si_snri = val_metrics["si_snri"]
                no_improve = 0
                torch.save(
                    _checkpoint(model, optimizer, epoch, global_step, cfg, best_si_snri),
                    ckpt_dir / "best.pt",
                )
                logger.log_artifact_file(str(ckpt_dir / "best.pt"))
            else:
                no_improve += 1
                if no_improve >= t["early_stop_patience"]:
                    print(f"[train] early stop at epoch {epoch} (best SI-SNRi {best_si_snri:.2f})")
                    break

        if (epoch + 1) % t["save_every_epochs"] == 0:
            torch.save(
                _checkpoint(model, optimizer, epoch, global_step, cfg, best_si_snri),
                ckpt_dir / f"epoch_{epoch}.pt",
            )

    torch.save(_checkpoint(model, optimizer, t["epochs"] - 1, global_step, cfg, best_si_snri),
               ckpt_dir / "last.pt")
    logger.finish()
    print(f"[train] done. best SI-SNRi {best_si_snri:.2f} dB")


# ---------------------------------------------------------------------------
def _load_manifest_or_build(d: dict, cfg: dict):
    """Load the train manifest; fall back to building from corpus roots."""
    from conformer_bimamba.data import (
        build_commonvoice_manifest,
        build_vivos_manifest,
        load_manifest,
    )

    path = d.get("train_manifest")
    if path and Path(path).exists():
        entries = load_manifest(path)
        if entries:
            return entries
        print(f"[data] manifest {path} empty; rebuilding from roots")
    entries = []
    if d.get("vivos_root"):
        entries += build_vivos_manifest(d["vivos_root"], "train", d.get("f0_root"))
    if d.get("commonvoice_root"):
        entries += build_commonvoice_manifest(d["commonvoice_root"], "train", d.get("f0_root"))
    if not entries:
        raise FileNotFoundError(
            "no training data: set data.train_manifest or data.vivos_root/commonvoice_root"
        )
    return entries


def _checkpoint(model, optimizer, epoch, step, cfg, best_si_snri) -> dict:
    return {
        "epoch": epoch, "step": step, "best_si_snri": best_si_snri,
        "model_state": copy.deepcopy(model.state_dict()),
        "optimizer_state": optimizer.state_dict(),
        "config": cfg,
    }


@torch.no_grad()
def _log_audio_sample(model, val_loader, logger, device, amp, step) -> None:
    batch = next(iter(val_loader))
    mixture = batch["mixture"].to(device)
    sources = batch["sources"].to(device)
    f0 = batch["f0"].squeeze(-1).to(device)
    with torch.autocast("cuda", enabled=amp):
        est = model(mixture, f0)
    T = min(est.shape[-1], sources.shape[-1])
    sr = int(os.environ.get("AUDIO_LOG_SR", "32000"))
    logger.log_audio("sample/mixture", mixture[0], sr, step)
    logger.log_audio("sample/ref_s1", sources[0, 0, :T], sr, step)
    logger.log_audio("sample/ref_s2", sources[0, 1, :T], sr, step)
    logger.log_audio("sample/est_s1", est[0, 0, 0, :T], sr, step)
    logger.log_audio("sample/est_s2", est[0, 1, 0, :T], sr, step)


if __name__ == "__main__":
    main()
