import json
from dataclasses import replace

import numpy as np
import pytest

from Pose.evaluation.experiments import accuracy
from Pose.evaluation.metrics import Track, contiguous_runs, lag, match_events
from Pose.evaluation.recording import FrameRecord, RecordingWriter, load_recording
from Pose.evaluation.replay import Analysis, replay, weighted_lag
from Pose.filtering import JointState
from Pose.motion import MotionEvent
from Pose.skeleton import JOINT_NAMES, NUM_JOINTS
from Pose.tracker import TrackerConfig

from .poses import raw, standing


def test_contiguous_runs():
    mask = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1], dtype=bool)
    assert contiguous_runs(mask, min_length=2) == [slice(1, 3), slice(4, 7)]


def lag_track(delay=0.0):
    rate = 30.0
    time = np.arange(300) / rate
    motion = lambda t: 50 * np.sin(2 * np.pi * 0.7 * t) + 20 * np.sin(2 * np.pi * 1.9 * t)
    reference = np.repeat(np.stack([motion(time), motion(time)], axis=1)[:, None], NUM_JOINTS, axis=1)
    position = np.repeat(np.stack([motion(time - delay), motion(time - delay)], axis=1)[:, None], NUM_JOINTS, axis=1)
    track = Track(
        id=0, frames=np.arange(300), time=time, detected=np.ones(300, bool), raw=reference,
        confidence=np.ones((300, NUM_JOINTS)), position=position,
        state=np.full((300, NUM_JOINTS), JointState.TRACKED), scale=np.full(300, 100.0), min_confidence=0.6,
    )
    return track, reference


@pytest.mark.parametrize("delay", [0.0, 0.05, 0.12])
def test_lag_recovers_a_known_delay(delay):
    track, reference = lag_track(delay)
    assert lag(track, reference) == pytest.approx(delay, abs=0.005)


@pytest.mark.parametrize("case", ["missing_reference", "stationary", "untracked", "missing_y", "short"])
def test_lag_is_unavailable_without_measurable_motion(case):
    track, reference = lag_track()
    if case == "missing_reference":
        reference[:] = np.nan
    elif case == "stationary":
        reference[:] = standing()
        track.position[:] = reference
    elif case == "untracked":
        track.state[:] = JointState.LOST
    elif case == "missing_y":
        reference[..., 1] = np.nan
    else:
        track = replace(track, time=track.time[:1], position=track.position[:1],
                        scale=track.scale[:1], state=track.state[:1])
        reference = reference[:1]
    assert np.isnan(lag(track, reference))


def test_weighted_lag_excludes_unavailable_tracks():
    track, reference = lag_track(0.12)
    still = np.zeros(len(track.time), dtype=bool)
    valid = Analysis(track, reference, still)
    missing = Analysis(track, np.full_like(reference, np.nan), still)
    stationary_track = replace(track, position=np.broadcast_to(standing(), track.position.shape).copy())
    stationary = Analysis(stationary_track, stationary_track.position.copy(), still)
    assert weighted_lag([valid, missing, stationary]) == pytest.approx(0.12, abs=0.005)
    assert np.isnan(weighted_lag([missing, stationary]))


def test_events_are_matched_by_kind_side_and_time():
    online = [MotionEvent("pull", "left", 1.05, 0.6), MotionEvent("pull", "right", 2.0, 0.6), MotionEvent("jump", None, 5.0, 2.5)]
    reference = [MotionEvent("pull", "left", 1.0, 0.6), MotionEvent("pull", "left", 3.0, 0.6)]
    match = match_events(online, reference)
    assert match.delays == pytest.approx([0.05])
    assert (match.extra, match.missed) == (2, 1)


def test_recording_round_trip_and_replay(tmp_path):
    path = tmp_path / "session.jsonl"
    rng = np.random.default_rng(1)
    with RecordingWriter(path, (640, 480), {"source": "test"}) as writer:
        for i in range(90):
            pose = raw(standing() + rng.normal(0, 1.5, (NUM_JOINTS, 2)))
            writer.write(FrameRecord(i, i / 30, [pose], detect_ms=20.0, brightness=120.0))
    recording = load_recording(path)
    assert recording.image_size == (640, 480) and len(recording.frames) == 90

    result = replay(recording, TrackerConfig())
    assert list(result.tracks) == [0]
    track = result.tracks[0]
    assert track.tracked[-1].all()
    assert np.std(track.position[30:, 0, 0]) < np.std(track.raw[30:, 0, 0])


@pytest.mark.parametrize("missed_poses", [[], [raw(standing(origin=(1000, 150)))]])
@pytest.mark.parametrize("all_missed", [False, True])
def test_accuracy_counts_unmatched_people_as_failures(tmp_path, missed_poses, all_missed):
    path = tmp_path / "session.jsonl"
    annotation_path = tmp_path / "annotations.json"
    pose = raw(standing())
    with RecordingWriter(path, (640, 480)) as writer:
        for i in range(4):
            poses = missed_poses if all_missed or i == 3 else [pose]
            writer.write(FrameRecord(i, i / 30, poses, 20.0, 120.0))
    person = dict(zip(JOINT_NAMES, pose.points.tolist()))
    person["right_ankle"] = None
    annotation_path.write_text(json.dumps({"frames": {"2": [person], "3": [person]}}), encoding="utf-8")

    result = accuracy(path, annotation_path, TrackerConfig())

    assert result["annotated_people"] == 2
    assert result["matched_people"] == (0 if all_missed else 1)
    assert set(result["joints"]) == set(JOINT_NAMES) - {"right_ankle"}
    for metrics in result["joints"].values():
        assert metrics["annotated"] == 2
        expected = 0.0 if all_missed else 0.5
        assert metrics["detection_rate"] == expected
        assert metrics["pck_raw"] == expected
        assert metrics["pck_filtered"] == expected
        for key in ("median_error_px", "mean_error_raw", "mean_error_filtered"):
            if all_missed:
                assert np.isnan(metrics[key])
            else:
                assert metrics[key] == pytest.approx(0.0)
