#!/usr/bin/env python3
"""
Datset and dataloader.

Authors:
LICENCE:
"""


from pathlib import Path
from typing import List, Tuple

import numpy as np
import torchvision.transforms as transforms
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class Enduro_Record(Dataset):
    """Enduro Dataset."""

    def __init__(
        self,
        base_path: Path,
        trials: List[str],
        target_transforms=None,
    ) -> None:
        """Ctor."""
        all_actions = []
        all_states = []

        for t in trials:
            a = np.load(base_path / t / "action_state.npz")
            all_actions.extend([*a["actions"]])
            all_states.extend([*a["states"]])

        self.actions = np.array(all_actions)
        self.states = np.array(all_states)

        self.transforms = target_transforms
        if self.transforms is None:

            def crop(img: np.array) -> np.array:
                """Crop frame to 160x160."""
                return img[:160, :, :]

            self.transforms = transforms.Compose(
                [
                    crop,
                    transforms.ToTensor(),
                ]
            )

    def __getitem__(self, idx) -> Tuple[Tensor, int]:
        """Get state and expected action at idx."""
        return self.transforms(self.states[idx]), self.actions[idx]

    def __len__(self) -> None:
        """Return len of datset."""
        return self.actions.size

    def loader(
        self,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 4,
    ) -> DataLoader:
        """Create a dataloader."""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
        )
