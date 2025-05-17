import matplotlib.pyplot as plt
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
