#!/usr/bin/env python3
"""
Record game.

Authors:
LICENCE:
"""

import os
from pathlib import Path
from time import sleep

import gym
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import games


class Record:
    """Record a game from the environment."""

    def __init__(
        self,
        game: games.Game_type,
        record: bool = False,
        store_path: Path = None,
    ) -> None:
        """Ctor."""
        self.game = game
        self.env = gym.make(game.name)
        self.env.reset()
        self.env.render(mode="human")
        self.record = record
        self.store_path = store_path
        if self.store_path is None:
            self.store_path = Path("temp/trial")
        if not os.path.exists(self.store_path):
            os.mkdir(self.store_path)
        if self.record:
            self.video = VideoRecorder(
                self.env,
                str(self.store_path / "video.mp4"),
            )
        self.env.viewer.window.push_handlers(game.keyboard)
        self.actions = []
        self.states = []

    def close(self) -> None:
        """Close game."""
        self.env.close()
        if self.record:
            self.video.close()

    def record_game(self) -> None:
        """Record a game for a given env."""
        isopen = True
        action_state_path = self.store_path / "action_state"
        while isopen:
            self.env.reset()
            total_reward = 0.0
            steps = 0
            restart = False
            while True:
                action = self.game.get_action()
                state, reward, done, _ = self.env.step(action)
                self.actions.append(action)
                self.states.append(state)
                total_reward += reward
                if steps % 200 == 0 or done:
                    print("\naction {:+0.2f}".format(action))
                    print(f"step {steps} total_reward {total_reward:+0.2f}")
                steps += 1
                if self.record:
                    self.video.capture_frame()
                isopen = self.env.render(mode="human")
                if done or restart or not isopen:
                    break
                sleep(0.08)
        if self.record:
            np.savez_compressed(
                action_state_path,
                actions=np.array(self.actions, dtype=np.uint8),
                states=np.array(self.states, dtype=np.uint8),
            )


def main() -> None:
    """Test record on Enduro-v4."""
    recenv = Record(games.Enduro(), record=True)
    recenv.record_game()
    recenv.close()


if __name__ == "__main__":
    main()
