#!/usr/bin/env python3
"""
Datset and dataloader.

Authors:
LICENCE:
"""


import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


class Enduro_Record(Dataset):
    """Enduro Dataset."""

    def __init__(self, path, target_transforms=None):
        """Ctor."""
        a = np.load(path / "action_state.npz")
        self.actions = a["actions"]
        self.states = a["states"]
        self.transforms = target_transforms
        if self.transforms is None:
            self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        """Get state and expected action at idx."""
        return self.transforms(self.states[idx]), self.actions[idx]

    def __len__(self):
        """Return len of datset."""
        return self.actions.size

    def loader(self, batch_size=64, shuffle=True, num_workers=4):
        """Create a dataloader."""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
        )
