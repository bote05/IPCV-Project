import json
import sys
from dataclasses import replace

import numpy as np
import pytest

from Pose.evaluation import __main__ as cli
from Pose.evaluation.experiments import JOINT_GROUPS, EvaluationError, accuracy, dropout, summarize, sweep
from Pose.evaluation.metrics import Track, acceleration_rms, contiguous_runs, lag, match_events, reference_trajectory, static_mask
from Pose.evaluation.recording import FrameRecord, RecordingWriter, load_recording
from Pose.evaluation.replay import Analysis, replay, weighted_lag
from Pose.evaluation.report import write_json
from Pose.filtering import JointState
from Pose.motion import MotionEvent
from Pose.skeleton import JOINT_NAMES, NUM_JOINTS
from Pose.tracker import TrackerConfig

from .poses import raw, standing


def test_contiguous_runs():
    mask = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1], dtype=bool)
    assert contiguous_runs(mask, min_length=2) == [slice(1, 3), slice(4, 7)]


def uneven_time(count=300):
    """Timestamps alternating 20 ms and 60 ms apart, as when the webcam reader drops frames."""
    return np.r_[0.0, np.cumsum(np.tile([0.02, 0.06], count)[: count - 1])]


def lag_track(delay=0.0, time=None):
    time = np.arange(300) / 30 if time is None else time
    count = len(time)
    motion = lambda t: 50 * np.sin(2 * np.pi * 0.7 * t) + 20 * np.sin(2 * np.pi * 1.9 * t)
    reference = np.repeat(np.stack([motion(time), motion(time)], axis=1)[:, None], NUM_JOINTS, axis=1)
    position = np.repeat(np.stack([motion(time - delay), motion(time - delay)], axis=1)[:, None], NUM_JOINTS, axis=1)
    track = Track(
        id=0, frames=np.arange(count), time=time, detected=np.ones(count, bool), raw=reference,
        confidence=np.ones((count, NUM_JOINTS)), position=position,
        state=np.full((count, NUM_JOINTS), JointState.TRACKED), scale=np.full(count, 100.0), min_confidence=0.6,
    )
    return track, reference


@pytest.mark.parametrize("delay", [0.0, 0.05, 0.12])
def test_lag_recovers_a_known_delay(delay):
    track, reference = lag_track(delay)
    assert lag(track, reference) == pytest.approx(delay, abs=0.005)


@pytest.mark.parametrize("delay", [0.0, 0.05, 0.12])
def test_lag_uses_elapsed_time_with_uneven_samples(delay):
    time = uneven_time()
    track, reference = lag_track(delay, time)
    assert lag(track, reference) == pytest.approx(delay, abs=0.005)


@pytest.mark.parametrize("acceleration", [0.0, 200.0])
def test_acceleration_uses_actual_sample_intervals(acceleration):
    time = uneven_time()
    track, _ = lag_track(time=time)
    points = np.broadcast_to(standing(), track.position.shape).copy()
    points[..., 0] += (100 * time + 0.5 * acceleration * time**2)[:, None]
    result = acceleration_rms(track, points, track.tracked, np.ones(len(time), bool))
    np.testing.assert_allclose(result, acceleration / 100, atol=1e-8)


def test_reference_filter_preserves_constant_speed_with_uneven_samples():
    time = uneven_time()
    track, _ = lag_track(time=time)
    track.raw = np.broadcast_to(standing(), track.raw.shape).copy()
    track.raw[..., 0] += (100 * time)[:, None]
    reference = reference_trajectory(track, cutoff=2.0)
    np.testing.assert_allclose(reference[20:-20], track.raw[20:-20], atol=0.2)


@pytest.mark.parametrize("gap_type", ["confidence", "time"])
def test_reference_filter_does_not_bridge_gaps(gap_type):
    track, _ = lag_track()
    track.raw[:] = standing()
    if gap_type == "confidence":
        track.confidence[145:155] = 0.0
        track.raw[155:, :, 0] += 200
    else:
        track.time[150:] += 2.0
        track.raw[150:, :, 0] += 200
    reference = reference_trajectory(track, cutoff=2.0)
    if gap_type == "confidence":
        assert np.isnan(reference[145:155]).all()
    valid = np.isfinite(reference)
    np.testing.assert_allclose(reference[valid], track.raw[valid], atol=1e-8)


@pytest.mark.parametrize(("count", "still"), [(32, True), (24, False)])  # 1.24 s and 0.92 s
def test_still_periods_are_measured_in_elapsed_time(count, still):
    track, _ = lag_track(time=uneven_time(count))
    reference = np.broadcast_to(standing(), track.raw.shape).copy()
    assert (static_mask(track, reference, min_duration=1.0) == still).all()


@pytest.mark.parametrize("case",["missing_reference", "stationary", "untracked", "missing_y", "short"])
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


def test_still_time_and_gap_lengths_use_elapsed_time_with_uneven_samples(tmp_path):
    path = tmp_path / "session.jsonl"
    time = uneven_time(150)
    with RecordingWriter(path, (640, 480)) as writer:
        for i, t in enumerate(time):
            writer.write(FrameRecord(i, float(t), [raw(standing())], detect_ms=20.0, brightness=120.0))

    summary, [analysis] = summarize(path, TrackerConfig(), [], cutoff=6.0)
    still = analysis.track.time[analysis.still]  # all frames once the torso length is known
    assert len(still) > 140
    assert summary["tracks"][0]["still_s"] == pytest.approx(still[-1] - still[0] + 0.02)

    # 5 hidden frames, two 20 ms and two 60 ms intervals plus the last frame's 20 ms
    rows = dropout(path, TrackerConfig(), JOINT_GROUPS["arms"], [0.1])
    assert [row["gap_ms"] for row in rows] == pytest.approx([180.0] * len(rows))


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


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("detected_x", [[200], [200, 240], [200, 800]])
def test_accuracy_matches_each_detection_to_only_one_person(tmp_path, reverse, detected_x):
    path = tmp_path / "session.jsonl"
    annotation_path = tmp_path / "annotations.json"
    poses = [raw(standing(origin=(x, 150))) for x in detected_x]
    with RecordingWriter(path, (640, 480)) as writer:
        for i in range(4):
            writer.write(FrameRecord(i, i / 30, poses, 20.0, 120.0))
    people = [dict(zip(JOINT_NAMES, standing(origin=(x, 150)).tolist())) for x in (200, 210)]
    if reverse:
        people.reverse()
    annotation_path.write_text(json.dumps({"frames": {"3": people}}), encoding="utf-8")

    result = accuracy(path, annotation_path, TrackerConfig())

    both_matched = 240 in detected_x
    assert result["annotated_people"] == 2
    assert result["matched_people"] == (2 if both_matched else 1)
    for metrics in result["joints"].values():
        assert metrics["annotated"] == 2
        assert metrics["detection_rate"] == (1.0 if both_matched else 0.5)
        assert metrics["pck_raw"] == 0.5
        assert metrics["pck_filtered"] == 0.5
        assert metrics["median_error_px"] == pytest.approx(15.0 if both_matched else 0.0)


def test_json_output_is_standard_with_unavailable_measurements_as_null(tmp_path):
    path = tmp_path / "out.json"
    data = {
        "lag_ms": np.nan,
        "fps_low": np.float64(np.inf),
        "joints": {"nose": {"annotated": np.int64(3), "pck_raw": np.float32(0.5), "seen": np.bool_(True)}},
        "per_joint": np.array([0.25, np.nan, -np.inf]),
        "image_size": (640, 480),
        "events": [{"median_delay_ms": None}],
    }
    write_json(str(path), data)

    def reject(constant):
        raise ValueError(f"nonstandard JSON constant {constant}")

    assert json.loads(path.read_text(encoding="utf-8"), parse_constant=reject) == {
        "lag_ms": None,
        "fps_low": None,
        "joints": {"nose": {"annotated": 3, "pck_raw": 0.5, "seen": True}},
        "per_joint": [0.25, None, None],
        "image_size": [640, 480],
        "events": [{"median_delay_ms": None}],
    }
    assert np.isnan(data["lag_ms"]) and np.isnan(data["per_joint"][1])


def test_unusable_recording_raises_and_only_the_command_line_exits(tmp_path, monkeypatch, capsys):
    path = tmp_path / "short.jsonl"
    with RecordingWriter(path, (640, 480)) as writer:
        for i in range(30):  # 1 s, shorter than any evaluated track
            writer.write(FrameRecord(i, i / 30, [raw(standing())], detect_ms=20.0, brightness=120.0))
    with pytest.raises(EvaluationError, match="no track"):
        sweep(path, TrackerConfig(), [], 6.0, [1.0], [1.0])

    monkeypatch.setattr(sys, "argv", ["evaluation", "sweep", str(path)])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 1
    assert "error: no track" in capsys.readouterr().err
