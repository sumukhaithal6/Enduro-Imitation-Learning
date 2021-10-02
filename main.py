#!/usr/bin/env python3
"""
RL Project.

UE18CS400SH.
Authors:
LICENCE:
"""


import argparse
from pathlib import Path

import torch

from datasets import Enduro_Record
from games import Enduro
from models import SimpleNet, ResNet18, BigNet
from train import trainer
from utils import model_play


def parse_arguments() -> argparse.Namespace:
    """Parse arguments from CLI."""
    parser = argparse.ArgumentParser(description="Enduro Learner")
    parser.add_argument("--store_path", type=Path, default="temp/")
    parser.add_argument("--trial_name", type=str, default="g1")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--model_path", type=Path, default="models/")
    parser.add_argument("--train_run_name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--opt", type=str, default="adam")
    parser.add_argument("--watch", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Learn Enduro."""
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {args.device}")

    model = ResNet18().to(args.device)
    # model = BigNet().to(args.device)

    if args.train:
        loader = Enduro_Record(args.store_path, args.trial_name).loader(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        if args.opt=="sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=args.learning_rate,
                momentum=0.9,
                nesterov=True
            )
        elif args.opt=="adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.learning_rate,
            )

        trainer(model, loader, optimizer, args)

    model.load_state_dict(
        torch.load(
            (args.model_path / args.train_run_name / "model.pth"),
            map_location=args.device,
        )
    )

    model_play(
        model,
        Enduro(),
        args,
    )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
