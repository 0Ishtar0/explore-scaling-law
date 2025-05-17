import torch
from torch import nn, Tensor
from torch import optim
from tqdm import tqdm

from data import TrainCurve


def huber(delta: float, r: Tensor) -> Tensor:
    return torch.where(torch.abs(r) < delta, 0.5 * r ** 2, delta * (torch.abs(r) - 0.5 * delta))


def compute_loss(model: nn.Module, data: TrainCurve) -> Tensor:
    pred = model(data)
    train_loss = data.losses
    r = torch.log(train_loss) - torch.log(pred.clamp(min=1e-10))
    return huber(0.001, r).sum()


def fit(model: nn.Module, lr: float, data: TrainCurve, max_steps: int) -> tuple[list[float], float]:
    optimizer = optim.SGD(model.parameters(), lr=lr)
    best_params = None
    best_loss = float('inf')

    for _ in tqdm(range(max_steps)):
        optimizer.zero_grad()
        loss = compute_loss(model, data)
        loss.backward()
        optimizer.step()

        if loss < best_loss:
            best_loss = loss.item()
            best_params = [p.item() for p in model.parameters()]

    return best_params, best_loss
