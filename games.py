#!/usr/bin/env python3
"""
Game details.

Authors:
LICENCE:
"""
try:
    from pyglet.window import key
except Exception:

    class key:
        """Import fails on colab."""

        LEFT = 0
        RIGHT = 0
        UP = 0
        DOWN = 0
        SPACE = 0

        class KeyStateHandler:
            """Type class."""

            pass


class Game_type:
    """Generic env definition."""

    def __init__(self, name) -> None:
        """Ctor."""
        self.sleep_time = 0
        self.name = name
        self.keyboard = key.KeyStateHandler()

    def get_action() -> int:
        """Return action based on keybaord state."""
        pass


class Enduro(Game_type):
    """Enduro game design."""

    def __init__(self) -> None:
        """Ctor."""
        super().__init__("Enduro-v4")
        self.sleep_time = 0.00

    def get_action(self) -> int:
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
        left = self.keyboard[key.LEFT]
        right = self.keyboard[key.RIGHT]
        down = self.keyboard[key.DOWN]
        fire = self.keyboard[key.SPACE]
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
