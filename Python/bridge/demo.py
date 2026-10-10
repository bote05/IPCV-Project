"""Fake two-player data for testing the bridge without a webcam."""

import math

from .protocol import Landmark, MotionSignal, PlayerState


def demo_players(elapsed_seconds: float) -> tuple[PlayerState, ...]:
    players = []
    for player_id, center_x in ((1, 0.3), (2, 0.7)):
        phase = elapsed_seconds * 2 + player_id
        pose = [Landmark((center_x, 0.5, 0.0), 1.0) for _ in range(33)]
        # Set a few joints so the debug panel shows something moving.
        for index, x, y in (
            (0, center_x, 0.2),
            (11, center_x - 0.06, 0.35),
            (12, center_x + 0.06, 0.35),
            (15, center_x - 0.12, 0.3 + 0.15 * math.sin(phase)),
            (16, center_x + 0.12, 0.3 + 0.15 * math.cos(phase)),
            (23, center_x - 0.04, 0.6),
            (24, center_x + 0.04, 0.6),
        ):
            pose[index] = Landmark((x, y, 0.0), 1.0)
        players.append(PlayerState(
            player_id, True, tuple(pose), (center_x - 0.05, 0.12, 0.1, 0.14),
            head_rotation_deg=(20 * math.sin(phase), 10 * math.cos(phase), 0),
            world_position_m=((player_id - 1.5) * 0.8, 0, 2),
            motion_signals=(
                MotionSignal("left_reach", math.sin(phase) < 0, 1),
                MotionSignal("right_reach", math.cos(phase) < 0, 1),
            ),
        ))
    return tuple(players)
