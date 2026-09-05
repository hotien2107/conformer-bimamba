"""Linear warmup + cosine-annealing LR schedule (Transformer-style)."""
from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineSchedule(LRScheduler):
    """Linear warmup for ``warmup_steps`` then cosine decay to ``min_lr``.

    Mirrors the schedule used by Sepformer / Conformer training.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 1e-3,
        last_epoch: int = -1,
    ):
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps = max(1, int(total_steps))
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            factor = (step + 1) / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, max(0.0, progress))
            factor = self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (
                1.0 + math.cos(math.pi * progress)
            )
        return [base_lr * factor for base_lr in self.base_lrs]
