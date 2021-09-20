#!/usr/bin/env python3
"""
RL Project.

UE18CS400SH.
Authors:
LICENCE:
"""


import argparse
from pathlib import Path

from datasets import Enduro_Record
from models import SimpleNet
from record import Enduro, Record


def parse_arguments() -> argparse.Namespace:
    """Parse arguments from CLI."""
    parser = argparse.ArgumentParser(description="Enduro Learner")
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--store_path", type=Path, default="temp/")
    parser.add_argument("--trial_name", type=str, default="1")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
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

    dataset = Enduro_Record(args.store_path / args.trial_name)
    model = SimpleNet()
    # for i in dataset.loader(batch_size=args.batch_size):
    #     with torch.no_grad():
    #         print(model(i[0]).argmax(axis=1))


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
