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
from models import BigNet, ResNet18, SimpleNet
from train import trainer
from utils import model_play


def parse_arguments() -> argparse.Namespace:
    """Parse arguments from CLI."""
    parser = argparse.ArgumentParser(
        description="Enduro Learner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--store_path",
        type=Path,
        default="temp/",
        help="Place to store videos.",
    )
    parser.add_argument(
        "--trial_names",
        nargs="+",
        help="List of trials to train on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Train batch size.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers",
    )
    parser.add_argument("--train", action="store_true")
    parser.add_argument(
        "--model_path",
        type=Path,
        default="models/",
        help="Parent directory to store models",
    )
    parser.add_argument(
        "--train_run_name",
        type=str,
        required=True,
        help="Name of the trial",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate, what else?",
    )
    parser.add_argument(
        "--opt",
        type=str,
        default="adam",
        help="Name of the optimizer",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Use optuna hyper parameter tuning.",
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--model",type=str,required=True,choices=["Big","Simple","ResNet"],help="Model architecture")
    return parser.parse_args()


def main() -> None:
    """Learn Enduro."""
    args = parse_arguments()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {args.device}")

    model_type = {"Big":BigNet,"Simple": SimpleNet, "ResNet": ResNet18}[args.model]

    if args.train:
        loader = Enduro_Record(args.store_path, args.trial_names).loader(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        trainer(args)(model_type, loader, args)
    model = model_type()
    model.load_state_dict(
        torch.load(
            (args.model_path / args.train_run_name / "model.pth"),
            map_location=args.device,
        )
    )
    if args.watch:
        model_play(
            model,
            Enduro(),
            args,
        )


if __name__ == "__main__":
    main()
