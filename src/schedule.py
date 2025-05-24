import csv

import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import logging
from torch import nn
from tqdm import tqdm


def optimize_lr_schedule_mpl(
        best_params: list,
        total_steps: int,
        peak_lr: float,
        min_lr: float,
        lr: float,
        max_steps: int,
        warmup: int,
        name: str,
        fig_folder: str = "../fig/",
        data_folder: str = "../data/"
):
    L0, A, alpha, B, C, beta, gamma = best_params
    delta = nn.Parameter(torch.zeros(total_steps - warmup, dtype=torch.float32), requires_grad=True)
    warmup_bias = 0.5 * peak_lr * warmup
    optimizer = torch.optim.Adam([delta], lr=lr)
    for _ in tqdm(range(max_steps), desc="Optimizing LR Schedule"):
        optimizer.zero_grad()
        eta = peak_lr - torch.cumsum(delta.clamp(min=0), dim=0)
        eta = torch.clamp(eta, min=min_lr)
        lr_sum = torch.cumsum(eta, dim=0) + warmup_bias
        lr_sum = torch.concatenate([torch.tensor([0]), lr_sum], dim=0)
        LD = torch.sum(delta * (1 - (1 + C * eta ** (-gamma) *
                       (lr_sum[-1] - lr_sum[:-1])) ** (-beta)))
        pred = L0 + A * lr_sum[-1] ** (-alpha) - B * LD
        pred.backward()
        optimizer.step()

        with torch.no_grad():
            delta.clamp_(min=0, max=peak_lr)
            eta = peak_lr - torch.cumsum(delta, dim=0)
            delta.masked_fill_(eta <= min_lr, 0)
            opt_lr = eta.detach().numpy()

    csv_file_path = os.path.join(data_folder, f"{name}_lr_schedule.csv")
    os.makedirs(data_folder, exist_ok=True)
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["step", "lr"])
        for step, lr_value in enumerate(opt_lr, start=warmup):
            writer.writerow([step, lr_value])

    plt.figure()
    plt.plot(np.arange(warmup, total_steps), opt_lr)
    plt.grid(True)
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title(f"Optimized Learning Rate Schedule ({name})")
    plt.savefig(os.path.join(fig_folder, f"{name}.png"))
    plt.close()
    return opt_lr
