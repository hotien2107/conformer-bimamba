"""Experiment tracking: unified WandB + TensorBoard wrapper."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch


class ExperimentLogger:
    """Thin wrapper around WandB and TensorBoard.

    ``enabled=False`` gives a no-op logger so training runs without any
    tracking backend installed.
    """

    def __init__(
        self,
        project: str = "conformer-bimamba",
        run_name: str | None = None,
        config: dict | None = None,
        log_dir: str = "runs",
        use_wandb: bool = True,
        use_tensorboard: bool = True,
    ):
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.wandb = None
        self.tb = None

        if use_wandb:
            try:
                import wandb

                run_name = run_name or "conformer-bimamba"
                wandb.init(project=project, name=run_name, config=config or {})
                self.wandb = wandb
            except Exception as exc:  # pragma: no cover
                print(f"[logger] wandb unavailable ({exc}); falling back to TB only")
                self.use_wandb = False

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(log_dir=str(self.log_dir / (run_name or "run")))
            except Exception as exc:  # pragma: no cover
                print(f"[logger] tensorboard unavailable ({exc})")
                self.use_tensorboard = False

    # -- scalars ----------------------------------------------------------
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.use_wandb and self.wandb is not None:
            self.wandb.log({tag: value}, step=step)
        if self.use_tensorboard and self.tb is not None:
            self.tb.add_scalar(tag, value, step)

    def log_dict(self, d: dict, step: int, prefix: str = "") -> None:
        for k, v in d.items():
            self.log_scalar(f"{prefix}/{k}" if prefix else k, float(v), step)

    # -- audio -------------------------------------------------------------
    def log_audio(self, tag: str, wav: torch.Tensor, sr: int, step: int) -> None:
        """Log (1, T) waveform."""
        wav = wav.detach().float().cpu()
        if self.use_wandb and self.wandb is not None:
            self.wandb.log({tag: self.wandb.Audio(wav.numpy(), sample_rate=sr)}, step=step)
        if self.use_tensorboard and self.tb is not None:
            self.tb.add_audio(tag, wav, step, sample_rate=sr)

    # -- artifacts ----------------------------------------------------------
    def log_config(self, config: dict) -> None:
        if self.use_wandb and self.wandb is not None:
            self.wandb.config.update(config, allow_val_change=True)

    def log_artifact_file(self, path: str) -> None:
        if self.use_wandb and self.wandb is not None and os.path.exists(path):
            art = self.wandb.Artifact(name=Path(path).stem, type="model")
            art.add_file(path)
            self.wandb.log_artifact(art)

    def finish(self) -> None:
        if self.use_wandb and self.wandb is not None:
            self.wandb.finish()
        if self.use_tensorboard and self.tb is not None:
            self.tb.close()
