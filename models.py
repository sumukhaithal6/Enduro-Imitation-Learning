#!/usr/bin/env python3
"""
Model architecture.

Authors:
LICENCE:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    """Test net."""

    def __init__(self):
        """Ctor."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(29008, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 9)

    def forward(self, x):
        """Forward call."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
