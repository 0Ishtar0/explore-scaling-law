import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from data import TrainCurve


def evaluate(data: TrainCurve, model: nn.Module, fig_folder: str):
    with torch.no_grad():
        pred = model(data)

    steps = data.steps.cpu().numpy()
    pred = pred.cpu().detach().numpy()
    losses = data.losses.cpu().numpy()
    plt.figure()
    plt.plot(steps, pred, label=f"{data.name}_pred", linestyle="--")
    plt.plot(steps, losses, label=f"{data.name}_loss", linestyle="-")
    plt.legend()
    plt.savefig(f"{fig_folder}/{data.name}_fit.png")
    plt.close()
    err = np.max(np.abs(pred - losses))
    print(f"|pred-loss|_1 for {data.name}: {err:.4e}")
    err = np.linalg.norm(pred - losses) / np.linalg.norm(losses)
    print(f"relative_error for {data.name}: {err:.4e}")
