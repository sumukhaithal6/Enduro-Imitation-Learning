#!/usr/bin/env python3
"""
Simple utilities.

Authors:
LICENCE:
"""

from argparse import Namespace

import gym
import torch
import torchvision.transforms as transforms
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from record import Game_type


def model_play(model: torch.nn.Module, game: Game_type, args: Namespace):
    """Make model play game and store video."""
    env = gym.make(game.name)
    video = VideoRecorder(
        env, str(args.model_path / args.trial_name / "train_video.mp4")
    )
    model.eval()
    data_transforms = transforms.ToTensor()
    cur_state = data_transforms(env.reset()).unsqueeze(0)
    total_reward = 0.0
    steps = 0
    while True:
        with torch.no_grad():
            a = model(cur_state.to(args.device)).cpu()
            a = a.argmax(dim=1)
        s, r, done, info = env.step(a)
        cur_state = data_transforms(s).unsqueeze(0)
        total_reward += r
        if steps % 200 == 0 or done:
            print("\naction ", a)
            print(f"step {steps} total_reward {total_reward:+0.2f}")
        steps += 1
        video.capture_frame()
        if args.watch:
            env.render(mode="human")
        if done:
            break
