import json

import numpy as np

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
