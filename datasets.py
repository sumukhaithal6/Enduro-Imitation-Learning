#!/usr/bin/env python3
"""
Datset and dataloader.

Authors:
LICENCE:
"""


import numpy as np
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


class Enduro_Record(Dataset):
    """Enduro Dataset."""

    def __init__(self, base_path, trial_names, target_transforms=None):
        """Ctor."""
        trials = trial_names.split(",")
        all_actions = []
        all_states = []

        for t in trials:
            a = np.load(os.path.join(base_path, t, "action_state.npz"))
            all_actions.extend([*a["actions"]])
            all_states.extend([*a["states"]])

        self.actions = np.array(all_actions)
        self.states = np.array(all_states)

        self.transforms = target_transforms
        if self.transforms is None:
            self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        """Get state and expected action at idx."""
        return self.transforms(self.crop(self.states[idx])), self.actions[idx]

    def __len__(self):
        """Return len of datset."""
        return self.actions.size
    
    def crop(self, img):
        return img[:160,:,:]

    def loader(self, batch_size=64, shuffle=True, num_workers=4):
        """Create a dataloader."""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
        )
