#!/usr/bin/env python3
"""
Python program to record a game.

Authors:
LICENCE:
"""


import argparse
from pathlib import Path

from games import Enduro
from record import Record


def parse_arguments() -> argparse.Namespace:
    """Parse arguments from CLI."""
    parser = argparse.ArgumentParser(description="Enduro Learner")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--store_path", type=Path, default="temp/")
    parser.add_argument("--trial_name", type=str, default="")
    return parser.parse_args()


def main() -> None:
    """Play Enduro."""
    args = parse_arguments()
    RecEnv = Record(
        Enduro(),
        record=args.record,
        store_path=args.store_path / args.trial_name,
    )
    RecEnv.record_game()
    RecEnv.close()


if __name__ == "__main__":
    main()
