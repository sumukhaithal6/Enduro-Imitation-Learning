#!/usr/bin/env python3
"""
Model train routines.

Authors:
LICENCE:
"""

import os
from argparse import Namespace

import torch
import torch.nn as nn


def trainer(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    args: Namespace,
):
    """Train model."""
    loss_fn = nn.CrossEntropyLoss()
    log_interval = 50
    if not os.path.exists(args.model_path / args.trial_name):
        os.mkdir(args.model_path / args.trial_name)
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}")

        for batch_idx, data in enumerate(dataloader):
            states, action = data
            states = states.to(args.device)
            action = action.to(args.device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            output = model(states)
            loss = loss_fn(output, action)

            if (batch_idx + 1) % log_interval == 0:
                print(f"Batch {batch_idx+1} Loss: {loss}")
            loss.backward()
            optimizer.step()
        torch.save(
            model.state_dict(),
            args.model_path / args.trial_name / "model.pth",
        )
