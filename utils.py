#!/usr/bin/env python3
"""
Simple utilities.

Authors:
LICENCE:
"""

import gym
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import torch
import torchvision.transforms as transforms


def model_play(model, args):
    env = gym.make("Enduro-v4")
    actions = []
    states = []
    isopen = True
    data_transforms = transforms.ToTensor()
    video = VideoRecorder(env, "video.mp4")
    
    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        cur_state = torch.rand((1,3,210,160))
        model.eval()
        while True:
            with torch.no_grad():
                a = model(cur_state.to(args.device)).cpu()
                a = torch.argmax(a, dim=1)
            s, r, done, info = env.step(a)
            cur_state = data_transforms(s).unsqueeze(0)
            actions.append(a)
            states.append(s)
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction ",a)
                print(f"step {steps} total_reward {total_reward:+0.2f}")
            steps += 1
            # if self.record:
            video.capture_frame()
            isopen = env.render(mode="human")
            if done or restart or not isopen:
                break
            # sleep(0.08)