import argparse
import os
from copy import deepcopy

import torch

from data import load_data
from model import MPL, LRA
from evaluate import evaluate
from optimize import fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data", help="Path to the data directory")
    parser.add_argument("--model", type=str, default="MPL", choices=["MPL", "LRA"], help="Model to use")
    parser.add_argument("--output", type=str, default="../fig", help="Path to the output directory")
    args = parser.parse_args()

    data1 = load_data(f"{args.data}/cosine_full.csv", 33907, 1e-3, "cosine", end_lr=1e-4)
    data2 = load_data(f"{args.data}/wsd_full.csv", 33907, 1e-3, "wsd", end_lr=1e-4, decay_step=27125)
    data3 = load_data(f"{args.data}/811_full.csv", 33907, 1e-3, "8-1-1", end_lr=1e-4, mid_lr=0.00031622776601683794)
    # data4 = load_data(f"{args.data}/transformerAdamConstantLR_5000.csv", 5000, 3e-4, "constant")
    data5 = load_data(f"{args.data}/transformerAdamStepLR_5000.csv", 5000, 3e-4, "step", step_size=100, gamma=0.99)
    data6 = load_data(f"{args.data}/transformerAdamLambdaLR_5000.csv", 5000, 3e-4, "lambda")
    # data7 = load_data(f"{args.data}/transformerAdamCosineAnnealingLR_5000.csv", 5000, 3e-4, "cosine", end_lr=0.0)

    # train_data = deepcopy(data5)
    # train_data.truncate(5000)
    train_data = deepcopy(data1)
    train_data.truncate(10000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "MPL":
        model = MPL()
    else:
        model = LRA()
    model.to(device)
    model.train()
    fit(model, 0.1, train_data, 1000)

    model.eval()
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    evaluate(data1, model, args.output)
    evaluate(data2, model, args.output)
    evaluate(data3, model, args.output)
    # evaluate(data4, model, args.output)
    evaluate(data5, model, args.output)
    evaluate(data6, model, args.output)
    # evaluate(data7, model, args.output)


if __name__ == "__main__":
    main()
