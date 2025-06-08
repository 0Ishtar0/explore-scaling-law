import argparse
import os
from copy import deepcopy

import torch

from data import load_data
from model import MPL, LRA, _MPL
from evaluate import evaluate
from optimize import fit
from schedule import optimize_lr_schedule_mpl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data", help="Path to the data directory")
    parser.add_argument("--model", type=str, default="_MPL",
                        choices=["MPL", "LRA", "_MPL"], help="Model to use")
    parser.add_argument("--output", type=str, default="../fig", help="Path to the output directory")
    args = parser.parse_args()

    data1 = load_data(f"{args.data}/cosine_full.csv", 33907, 1e-3, "cosine", end_lr=1e-4)
    data2 = load_data(f"{args.data}/wsd_full.csv", 33907, 1e-3,
                      "wsd", end_lr=1e-4, decay_step=27125)
    data3 = load_data(f"{args.data}/811_full.csv", 33907, 1e-3, "8-1-1",
                      end_lr=1e-4, mid_lr=0.00031622776601683794)

    train_data = deepcopy(data1)
    train_data.truncate(10000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "MPL":
        model = MPL()
    elif args.model == "_MPL":
        model = _MPL()
    else:
        model = LRA()
    model = model.to(device)
    model.train()
    best_params, _ = fit(model, 0.1, train_data, 20000)
    print(best_params)
    # if args.model == "MPL":
    #     optimize_lr_schedule_mpl(best_params, 33907, 1e-3, 1e-4, 5e-9, 5000, 2000, "opt")

    model.eval()
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    evaluate(data1, model, args.output)
    evaluate(data2, model, args.output)
    evaluate(data3, model, args.output)


if __name__ == "__main__":
    main()
