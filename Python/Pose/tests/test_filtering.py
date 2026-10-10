import numpy as np
import pytest

from Pose.filtering import FilterConfig, JointState, KeypointFilter, OneEuroFilter

DT = 1 / 30
SCALE = 100.0


def run(filter_: KeypointFilter, points, confidences, start: float = 0.0) -> list[np.ndarray]:
    positions = []
    for i, (p, c) in enumerate(zip(points, confidences)):
        filter_.update(None if p is None else np.atleast_2d(p), None if c is None else np.atleast_1d(c), start + i * DT, SCALE)
        positions.append(filter_.position.copy())
    return positions


def test_one_euro_removes_most_jitter_at_rest():
    rng = np.random.default_rng(0)
    euro = OneEuroFilter((1, 2), min_cutoff=1.0, beta=1.0, d_cutoff=1.0)
    rows = np.ones(1, dtype=bool)
    euro.set(rows, np.zeros((1, 2)))
    samples = rng.normal(0.0, 2.0, (300, 1, 2))
    output = []
    for sample in samples:
        euro.update(sample, DT, rows, scale=SCALE)
        output.append(euro.value.copy())
    assert np.std(output[30:]) < 0.35 * np.std(samples)


@pytest.mark.parametrize("beta", [0.0, 2.0])
def test_one_euro_lag_shrinks_with_beta(beta):
    euro = OneEuroFilter((1, 2), min_cutoff=1.0, beta=beta, d_cutoff=1.0)
    rows = np.ones(1, dtype=bool)
    euro.set(rows, np.zeros((1, 2)))
    speed = 3 * SCALE  # 3 torso lengths/s
    for i in range(1, 60):
        euro.update(np.array([[speed * i * DT, 0.0]]), DT, rows, scale=SCALE)
    lag_seconds = (speed * 59 * DT - euro.value[0, 0]) / speed
    if beta == 0.0:
        assert lag_seconds > 0.12
    else:
        assert lag_seconds < 0.05


def test_joint_is_trusted_only_after_consistent_detections():
    config = FilterConfig(acquire_frames=3)
    f = KeypointFilter(1, config)
    states = []
    for i in range(4):
        f.update(np.array([[100.0, 100.0]]), np.array([0.9]), i * DT, SCALE)
        states.append(f.state[0])
    assert states == [JointState.ACQUIRING, JointState.ACQUIRING, JointState.TRACKED, JointState.TRACKED]


def test_confidence_hysteresis():
    f = KeypointFilter(1, FilterConfig(confidence_on=0.6, confidence_off=0.4, acquire_frames=1))
    run(f, [(100.0, 100.0)] * 2 + [(100.0, 100.0)] * 5, [0.5] * 2 + [0.9] + [0.5] * 4)
    assert f.state[0] == JointState.TRACKED  # 0.5 cannot start tracking but does not stop it


def test_short_gap_is_predicted_then_lost():
    config = FilterConfig(max_gap=0.2, acquire_frames=1)
    f = KeypointFilter(1, config)
    run(f, [(100.0, 100.0)] * 10, [0.9] * 10)
    states = []
    for i in range(10, 20):
        f.update(None, None, i * DT, SCALE)
        states.append(f.state[0])
    gap_frames = int(config.max_gap / DT)
    assert all(s == JointState.PREDICTED for s in states[:gap_frames])
    assert states[-1] == JointState.LOST


def test_prediction_continues_motion_with_decaying_velocity():
    config = FilterConfig(acquire_frames=1, prediction_decay=0.1, max_gap=1.0)
    f = KeypointFilter(1, config)
    speed = 2 * SCALE
    run(f, [(speed * i * DT, 0.0) for i in range(40)], [0.9] * 40)
    last_seen = f.position[0, 0]
    velocity = f.velocity[0, 0]
    for i in range(40, 70):
        f.update(None, None, i * DT, SCALE)
    travelled = f.position[0, 0] - last_seen
    assert 0 < travelled <= velocity * config.prediction_decay * 1.05


def test_hold_strategy_freezes_position():
    f = KeypointFilter(1, FilterConfig(acquire_frames=1, prediction_decay=0.0))
    run(f, [(i * 5.0, 0.0) for i in range(30)], [0.9] * 30)
    frozen = f.position.copy()
    for i in range(30, 35):
        f.update(None, None, i * DT, SCALE)
    np.testing.assert_allclose(f.position, frozen)


def test_single_frame_glitch_is_rejected():
    f = KeypointFilter(1, FilterConfig(acquire_frames=1))
    run(f, [(100.0, 100.0)] * 20, [0.9] * 20)
    f.update(np.array([[100.0 + 3 * SCALE, 100.0]]), np.array([0.9]), 20 * DT, SCALE)
    assert f.state[0] == JointState.PREDICTED
    assert abs(f.position[0, 0] - 100.0) < 1.0


def test_lost_joint_restarts_at_new_detection():
    f = KeypointFilter(1, FilterConfig(acquire_frames=2, max_gap=0.1))
    run(f, [(100.0, 100.0)] * 10, [0.9] * 10)
    run(f, [None] * 10, [None] * 10, start=10 * DT)
    assert f.state[0] == JointState.LOST
    f.update(np.array([[500.0, 100.0]]), np.array([0.9]), 20 * DT, SCALE)
    assert f.state[0] == JointState.ACQUIRING
    np.testing.assert_allclose(f.position[0], (500.0, 100.0))
