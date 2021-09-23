#!/usr/bin/env python3
"""
Record game.

Authors:
LICENCE:
"""

from pathlib import Path
from time import sleep
import os

import gym
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder

try:
    from pyglet.window import key
except Exception:

    class key:
        """Import fails on colab."""

        class KeyStateHandler:
            """Type class."""

            pass


class Game_type:
    """Generic env definition."""

    def __init__(self, name) -> None:
        """Ctor."""
        self.sleep_time = 0
        self.name = name

    def get_action(keyboard: key.KeyStateHandler) -> int:
        """Return action based on keybaord state."""
        pass


class Enduro(Game_type):
    """Enduro game design."""

    def __init__(self) -> None:
        """Ctor."""
        super().__init__("Enduro-v4")
        self.sleep_time = 0.00

    def get_action(self, keyboard: key.KeyStateHandler) -> int:
        """
        Get action from keyboard.

        # case PLAYER_A_NOOP:0
        # case PLAYER_A_FIRE:1
        # case PLAYER_A_RIGHT:2
        # case PLAYER_A_LEFT:3
        # case PLAYER_A_DOWN:4
        # case PLAYER_A_DOWNRIGHT:5
        # case PLAYER_A_DOWNLEFT:6
        # case PLAYER_A_RIGHTFIRE:7
        # case PLAYER_A_LEFTFIRE:8
        """
        left = keyboard[key.LEFT]
        right = keyboard[key.RIGHT]
        down = keyboard[key.DOWN]
        fire = keyboard[key.SPACE]
        a = 0
        if left:
            a += 3
        elif right:
            a += 2
        elif fire:
            return 1
        elif down:
            return 4
        if down:
            a += 3
        elif fire:
            a += 5
        return a


class Record:
    """Record a game from the environment."""

    def __init__(
        self,
        game: Game_type,
        record: bool = False,
        store_path: Path = None,
    ) -> None:
        """Ctor."""
        self.game = game
        self.env = gym.make(game.name)
        self.env.reset()
        self.record = record
        self.store_path = store_path
        if self.store_path is None:
            self.store_path = Path("temp/trial")
        if not os.path.exists(self.store_path):
            os.mkdir(self.store_path)
        self.action_state_path = self.store_path / "action_state"
        if self.record:
            self.video = VideoRecorder(
                self.env,
                str(self.store_path / "video.mp4"),
            )
        self.keyboard = key.KeyStateHandler()
        self.env.viewer.window.push_handlers(self.keyboard)
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
        while isopen:
            self.env.reset()
            total_reward = 0.0
            steps = 0
            restart = False
            while True:
                a = self.game.get_action(self.keyboard)
                s, r, done, info = self.env.step(a)
                self.actions.append(a)
                self.states.append(s)
                total_reward += r
                if steps % 200 == 0 or done:
                    print("\naction {:+0.2f}".format(a))
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
                self.action_state_path,
                actions=np.array(self.actions, dtype=np.uint8),
                states=np.array(self.states, dtype=np.uint8),
            )


def main(args) -> None:
    """Test record on Enduro-v4."""
    RecEnv = Record(Enduro(), record=True)
    RecEnv.record_game()
    RecEnv.close()


if __name__ == "__main__":
    main(None)
