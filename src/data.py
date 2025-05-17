from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor


@dataclass
class TrainCurve:
    name: str
    steps: Tensor
    losses: Tensor
    learning_rates: Tensor
    lr_sum: Tensor
    S1: Tensor
    lr_gap: Tensor
    N: int

    def truncate(self, max_steps: int) -> None:
        mask = self.steps < max_steps
        self.steps = self.steps[mask]
        self.losses = self.losses[mask]
        self.learning_rates = self.learning_rates[:max_steps]
        self.lr_sum = self.lr_sum[:max_steps]
        self.S1 = self.S1[mask]
        self.lr_gap = self.lr_gap[:max_steps]


def _cosine_lr(total: int, peak_lr: float, end_lr: float) -> NDArray:
    step = np.arange(total)
    cosine_lrs = end_lr + 0.5 * (peak_lr - end_lr) * (1 + np.cos(np.pi * step / total))
    return cosine_lrs


def _wsd_lrs(total: int, decay: int, peak_lr: float, end_lr: float) -> NDArray:
    step = np.arange(total)
    decay_lrs = peak_lr ** ((total - step) / (total - decay)) * end_lr ** ((step - decay) / (total - decay))
    return np.concatenate((np.full(decay, peak_lr), decay_lrs[decay:]))


def _eight_one_one_lrs(total: int, peak_lr: float, mid_lr: float, end_lr: float) -> NDArray:
    stage1_steps = int(total * 0.8)
    stage2_steps = int(total * 0.1)
    stage3_steps = total - stage1_steps - stage2_steps
    stage1_lrs = np.full(stage1_steps, peak_lr)
    stage2_lrs = np.full(stage3_steps, mid_lr)
    stage3_lrs = np.full(stage2_steps, end_lr)
    return np.concatenate((stage1_lrs, stage2_lrs, stage3_lrs))


def load_data(file_path: str, total_steps: int, peak_lr: float, end_lr: float,
              lr_type: Literal["cosine", "8-1-1", "wsd"],
              decay_step: Optional[int] = None,
              mid_lr: Optional[float] = None) -> TrainCurve:
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    steps = data[:, 0].astype(int)
    losses = data[:, 2].astype(float)

    if lr_type == "cosine":
        learning_rates = _cosine_lr(total_steps, peak_lr, end_lr)
    elif lr_type == "8-1-1":
        learning_rates = _eight_one_one_lrs(total_steps, peak_lr, mid_lr, end_lr)
    elif lr_type == "wsd":
        learning_rates = _wsd_lrs(total_steps, decay_step, peak_lr, end_lr)
    else:
        raise ValueError(f"Invalid learning rate type: {lr_type}")

    lr_sum = np.cumsum(learning_rates)
    S1 = lr_sum[steps]
    lr_gap = np.zeros_like(learning_rates)
    lr_gap[1:] = np.diff(learning_rates)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = torch.tensor(steps, device=device)
    losses = torch.tensor(losses, device=device)
    learning_rates = torch.tensor(learning_rates, device=device)
    lr_sum = torch.tensor(lr_sum, device=device)
    S1 = torch.tensor(S1, device=device)
    lr_gap = torch.tensor(lr_gap, device=device)

    return TrainCurve(name=lr_type, steps=steps, losses=losses, learning_rates=learning_rates, lr_sum=lr_sum, S1=S1,
                      lr_gap=lr_gap, N=int(1e8))
