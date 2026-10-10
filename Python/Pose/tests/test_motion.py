import numpy as np

from Pose.filtering import JointState
from Pose.motion import MotionAnalyzer
from Pose.skeleton import NUM_JOINTS, Joint

from .poses import standing

DT = 1 / 30
TRACKED = np.full(NUM_JOINTS, JointState.TRACKED)


def play(frames: list[np.ndarray], states=None, analyzer=None):
    """Feed positions to an analyzer with finite-difference velocities; return all signals."""
    analyzer = analyzer or MotionAnalyzer(NUM_JOINTS)
    signals = []
    for i, position in enumerate(frames):
        velocity = (position - frames[i - 1]) / DT if i else np.zeros_like(position)
        state = TRACKED if states is None else states[i]
        signals.append(analyzer.update(position, velocity, state, i * DT))
    return signals


def events(signals, kind):
    return [e for s in signals for e in s.events if e.kind == kind]


def wrist_stroke(top: float, bottom: float, duration: float) -> list[np.ndarray]:
    """Left wrist held at `top`, moved linearly down to `bottom` over `duration`, then held."""
    steps = round(duration / DT)
    heights = [top] * 10 + list(np.linspace(top, bottom, steps)) + [bottom] * 10
    return [standing(left_wrist=(0.5, -h)) for h in heights]


def test_body_frame_does_not_depend_on_position_or_distance():
    near = play([standing(origin=(200, 100), torso=150)] * 3)[-1]
    far = play([standing(origin=(500, 300), torso=60)] * 3)[-1]
    np.testing.assert_allclose(near.body, far.body, atol=1e-9)
    np.testing.assert_allclose(near.body[Joint.LEFT_WRIST], (0.5, -0.9))


def test_fast_downward_stroke_fires_one_pull():
    pulls = events(play(wrist_stroke(top=1.0, bottom=0.0, duration=0.3)), "pull")
    assert len(pulls) == 1
    assert pulls[0].side == "left"


def test_slow_lowering_is_not_a_pull():
    assert not events(play(wrist_stroke(top=1.0, bottom=0.0, duration=3.0)), "pull")


def test_no_pull_while_wrist_is_only_predicted():
    frames = wrist_stroke(top=1.0, bottom=0.0, duration=0.3)
    states = [TRACKED.copy() for _ in frames]
    for state in states[8:]:
        state[Joint.LEFT_WRIST] = JointState.PREDICTED
    assert not events(play(frames, states), "pull")


def test_hand_raised_uses_hysteresis():
    heights = [0.0, 0.7, 0.5, 0.3]  # above raise_on, between thresholds, below raise_off
    signals = play([standing(left_wrist=(0.5, -h)) for h in heights])
    assert [bool(s.hand_raised[0]) for s in signals] == [False, True, True, False]


def test_jump_fires_once_when_whole_body_rises_quickly():
    rising = [standing(origin=(320, 150 - 300 * DT * i)) for i in range(1, 7)]  # 3 torso lengths/s
    assert len(events(play([standing()] * 5 + rising), "jump")) == 1


def test_walking_away_from_camera_is_not_a_jump():
    # Seen from a camera above hip height the body rises in the image, but it also shrinks.
    receding = [standing(origin=(320, 150 - 300 * DT * i), torso=100 * 0.97**i) for i in range(1, 7)]
    assert not events(play([standing()] * 5 + receding), "jump")


def test_lost_shoulders_reset_discrete_outputs():
    frames = [standing(left_wrist=(0.5, -1.0))] * 3
    states = [TRACKED.copy() for _ in frames]
    for joint in (Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER):
        states[2][joint] = JointState.LOST
    signals = play(frames, states)
    assert signals[1].hand_raised[0] and signals[1].valid
    assert not signals[2].hand_raised[0] and not signals[2].valid
