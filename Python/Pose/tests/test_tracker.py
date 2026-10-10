import json
from collections import defaultdict

import numpy as np
import pytest

from Pose.filtering import JointState
from Pose.tracker import PoseTracker, TrackerConfig

from .poses import raw, standing

DT = 1 / 30
LEFT, RIGHT = standing(origin=(200, 150)), standing(origin=(450, 150))


def test_tracks_follow_people_regardless_of_detection_order():
    tracker = PoseTracker()
    ids_by_side = {"left": set(), "right": set()}
    for i in range(30):
        poses = [raw(LEFT), raw(RIGHT)] if i % 2 else [raw(RIGHT), raw(LEFT)]
        for player in tracker.update(poses, i * DT):
            side = "left" if player.raw.points[0, 0] < 320 else "right"
            ids_by_side[side].add(player.id)
    assert len(ids_by_side["left"]) == 1 and len(ids_by_side["right"]) == 1
    assert ids_by_side["left"] != ids_by_side["right"]


def test_duplicate_detections_of_one_person_become_one_track():
    tracker = PoseTracker()
    players = tracker.update([raw(LEFT), raw(LEFT + 2.0, confidence=0.8)], 0.0)
    assert len(players) == 1


def test_undetected_track_coasts_then_expires():
    tracker = PoseTracker(TrackerConfig(track_timeout=0.5))
    for i in range(10):
        tracker.update([raw(LEFT)], i * DT)
    last_seen = 9 * DT
    coasting = tracker.update([], last_seen + DT)
    assert len(coasting) == 1 and not coasting[0].detected
    assert (coasting[0].state == JointState.PREDICTED).all()
    assert len(tracker.update([], last_seen + 0.45)) == 1
    assert tracker.update([], last_seen + 0.55) == []


def test_external_ids_override_association():
    tracker = PoseTracker()
    tracker.update([raw(LEFT), raw(RIGHT)], 0.0, ids=[7, 3])
    players = tracker.update([raw(RIGHT), raw(LEFT)], DT, ids=[3, 7])
    by_id = {p.id: p for p in players}
    assert set(by_id) == {3, 7}
    assert by_id[7].raw.points[0, 0] < 320


def test_identity_module_can_discard_detections():
    tracker = PoseTracker()
    players = tracker.update([raw(LEFT), raw(RIGHT)], 0.0, ids=[None, 1])
    assert [p.id for p in players] == [1]
    assert players[0].raw.points[0, 0] > 320


@pytest.mark.parametrize("ids", [[1, 1], [1]])
def test_invalid_ids_are_rejected(ids):
    with pytest.raises(ValueError):
        PoseTracker().update([raw(LEFT), raw(RIGHT)], 0.0, ids=ids)


def pull_stroke(origin: tuple[float, float]) -> list[np.ndarray]:
    """Left hand held above the head, then pulled down to the chest in 0.2 s."""
    up = standing(origin=origin, left_wrist=(0.5, -1.0))
    down = standing(origin=origin, left_wrist=(0.5, 0.3))
    return [up] * 20 + [up + (down - up) * k / 6 for k in range(1, 7)] + [down] * 20


def test_events_are_repeated_for_event_hold_and_have_unique_ids():
    tracker = PoseTracker(TrackerConfig(event_hold=0.2))
    fired_at, seen_at = {}, defaultdict(list)
    for i, (left, right) in enumerate(zip(pull_stroke((200, 150)), pull_stroke((450, 150)))):
        t = i * DT
        for player in tracker.update([raw(left), raw(right)], t):
            fired_at.update({event.id: t for event in player.motion.events})
            for event in player.events:
                seen_at[event.id].append(t)
    assert len(fired_at) == 2
    for event_id, t in fired_at.items():
        assert seen_at[event_id][0] == t
        assert len(seen_at[event_id]) >= 5
        assert seen_at[event_id][-1] - t < 0.2 + 1e-9


def test_mirrored_message_flips_horizontal_values():
    tracker = PoseTracker()
    tilted = standing(left_shoulder=(0.35, -0.1), left_wrist=(0.8, -0.2))
    for i in range(5):
        player = tracker.update([raw(tilted)], i * DT)[0]
    plain, mirrored = player.to_dict((640, 480)), player.to_dict((640, 480), mirror=True)
    for a, b in zip(plain["joints"], mirrored["joints"]):
        assert b["name"] == a["name"] and b["y"] == a["y"]
        assert b["x"] == pytest.approx(1 - a["x"], abs=1e-4)
        assert b["body"] == pytest.approx([-a["body"][0], a["body"][1]])
    assert mirrored["center"][0] == pytest.approx(1 - plain["center"][0], abs=1e-4)
    assert plain["lean"] != 0 and mirrored["lean"] == pytest.approx(-plain["lean"])


def test_message_is_valid_json_with_normalised_coordinates():
    tracker = PoseTracker()
    for i in range(5):
        players = tracker.update([raw(LEFT)], i * DT)
    players = tracker.update([], 5 * DT)
    message = players[0].to_dict((640, 480))
    text = json.dumps(message, allow_nan=False)
    assert json.loads(text)["joints"][0]["name"] == "nose"
    xs = [joint["x"] for joint in message["joints"]]
    assert all(0.0 <= x <= 1.0 for x in xs)
    np.testing.assert_allclose(message["center"][0], LEFT[7:9, 0].mean() / 640, atol=1e-3)
