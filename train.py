#!/usr/bin/env python3
"""
Model train routines.

Authors:
LICENCE:
"""

import os
from argparse import Namespace

import optuna
import torch
from optuna.trial import TrialState
from torch import nn


def _train_loop(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    args: Namespace,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.CrossEntropyLoss,
) -> float:
    model.train()
    epoch_loss = 0.0
    log_interval = len(dataloader) // 4
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
        epoch_loss += loss.item()
    return epoch_loss


def _get_optimizer(model: nn.Module, args: Namespace) -> torch.optim.Optimizer:
    if args.opt == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            nesterov=True,
        )
    elif args.opt == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
        )
    return optimizer


def trainer(
    args: Namespace,
):
    """Return function."""
    return tune if args.tune else train


def tune(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    args: Namespace,
):
    """Tune hp."""

    def objective(trial):
        """Minimize train loss."""
        args.lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        loss = 0.0
        loss_fn = nn.CrossEntropyLoss()
        optimizer = _get_optimizer(model, args)
        for epoch in range(1, args.epochs + 1):
            print(f"Epoch: {epoch}")
            loss += _train_loop(model, dataloader, args, optimizer, loss_fn)
            trial.report(loss, epoch)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return loss

    study = optuna.create_study(direction=("minimize"))
    study.optimize(objective, n_trials=(100))

    pruned_trials = study.get_trials(
        deepcopy=False,
        states=[TrialState.PRUNED],
    )
    complete_trials = study.get_trials(
        deepcopy=False,
        states=[TrialState.COMPLETE],
    )

    print("Study statistics: ")
    print("\tNumber of finished trials: ", len(study.trials))
    print("\tNumber of pruned trials: ", len(pruned_trials))
    print("\tNumber of complete trials: ", len(complete_trials))
    print("Best trial:")
    trial = study.best_trial
    print("\tValue: ", trial.value)
    print("\tParams: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")


def train(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    args: Namespace,
):
    """Train model."""
    loss_fn = nn.CrossEntropyLoss()
    if not os.path.exists(args.model_path / args.train_run_name):
        os.mkdir(args.model_path / args.train_run_name)
    optimizer = _get_optimizer(model, args)
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch: {epoch}")
        _train_loop(model, dataloader, args, optimizer, loss_fn)
        torch.save(
            model.state_dict(),
            args.model_path / args.train_run_name / "model.pth",
        )
