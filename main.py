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
from models import SimpleNet
from record import Enduro, Record
from train import trainer
from utils import model_play


def parse_arguments() -> argparse.Namespace:
    """Parse arguments from CLI."""
    parser = argparse.ArgumentParser(description="Enduro Learner")
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--store_path", type=Path, default="temp/")
    parser.add_argument("--trial_name", type=str, default="1")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--model_path", type=Path, default="models/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--watch", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Learn Enduro."""
    if args.play or args.record:
        RecEnv = Record(
            Enduro(),
            record=args.record,
            store_path=args.store_path / args.trial_name,
        )
        RecEnv.record_game()
        RecEnv.close()
        return

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on: {}".format(args.device))

    dataset = Enduro_Record(args.store_path / args.trial_name)
    loader = dataset.loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = SimpleNet().to(args.device)
    if args.train:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
        )
        trainer(model, loader, optimizer, args)
    model.load_state_dict(torch.load((args.model_path / args.trial_name / "model.pth")))
    model_play(
        model,
        Enduro(),
        args,
    )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
