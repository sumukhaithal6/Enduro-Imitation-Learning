#!/usr/bin/env python3
"""
RL Project.

UE18CS400SH.
Authors:
LICENCE:
"""


import argparse
from pathlib import Path

from record import Enduro, Record


def parse_arguments() -> argparse.Namespace:
    """Parse arguments from CLI."""
    parser = argparse.ArgumentParser(description="Enduro Learner")
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--store_path", type=Path, default="temp/")
    parser.add_argument("--trial_name", type=str, default="1")
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


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
