import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from data import TrainCurve
from model import MPL, LRA


def evaluate(data: TrainCurve, model: nn.Module, fig_folder: str):
    with torch.no_grad():
        if isinstance(model, MPL):
            pred = model(data)
        elif isinstance(model, LRA):
            pred = model.forward_low_mem(data)
        else:
            raise ValueError("Unknown model type")

    steps = data.steps.cpu().numpy()
    pred = pred.cpu().detach().numpy()
    losses = data.losses.cpu().numpy()
    plt.figure()
    plt.plot(steps, pred, label=f"{data.name}_pred", linestyle="--")
    plt.plot(steps, losses, label=f"{data.name}_loss", linestyle="-")
    plt.legend()
    plt.savefig(f"{fig_folder}/{data.name}_fit.pdf")
    plt.close()
    err = np.mean(np.abs(pred - losses))
    print(f"|pred-loss|_1 for {data.name}: {err:.4e}")
    err = np.linalg.norm(pred - losses) / np.linalg.norm(losses)
    print(f"relative error for {data.name}: {err:.4e}")
