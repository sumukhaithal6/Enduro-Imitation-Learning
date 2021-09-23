#!/usr/bin/env python3
"""
Simple utilities.

Authors:
LICENCE:
"""

from argparse import Namespace
from time import sleep

import gym
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from torchvision import transforms

from games import Game_type


def model_play(model: torch.nn.Module, game: Game_type, args: Namespace):
    """Make model play game and store video."""
    env = gym.make(game.name)
    video = VideoRecorder(
        env,
        str(
            args.model_path / args.train_run_name / (args.train_run_name + ".mp4"),
        ),
    )
    model.eval()
    data_transforms = transforms.ToTensor()
    cur_state = data_transforms(env.reset()).unsqueeze(0)
    total_reward = 0.0
    steps = 0
    while True:
        with torch.no_grad():
            action = model(cur_state.to(args.device)).cpu()
            action = action.argmax(dim=1)
        state, reward, done, _ = env.step(action)
        cur_state = data_transforms(state).unsqueeze(0)
        total_reward += reward
        if steps % 200 == 0 or done:
            print("\naction ", action)
            print(f"step {steps} total_reward {total_reward:+0.2f}")
        steps += 1
        video.capture_frame()
        if args.watch:
            isopen = env.render(mode="human")
            sleep(0.01)
            if not isopen:
                break
        if done:
            break
