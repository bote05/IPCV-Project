"""Run with Pose's branch and the shared requirements installed; no camera or model needed."""

import importlib.util
import json
import unittest

from bridge.pipeline import player_from_pose
from bridge.protocol import Joint, JointState, encode_frame

POSE_AVAILABLE = importlib.util.find_spec("Pose.tracker") is not None
if POSE_AVAILABLE:
    import numpy as np
    from Pose import PoseTracker, RawPose
    from Pose.skeleton import JOINT_NAMES


@unittest.skipUnless(POSE_AVAILABLE, "Pose branch has not been merged or added to PYTHONPATH")
class PoseIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tracker = PoseTracker()
        self.points = np.array([
            (300, 100), (350, 150), (250, 150), (390, 180), (210, 180),
            (350, 40), (200, 250), (330, 250), (270, 250),
            (330, 350), (270, 350), (330, 450), (270, 450),
        ], dtype=float)

    def raw(self, points=None):
        return RawPose(self.points.copy() if points is None else points, np.ones(13))

    def test_identity_order_mirroring_and_serialization(self):
        self.assertEqual(JOINT_NAMES, tuple(j.name.lower() for j in Joint))
        for i in range(5):
            players = self.tracker.update([self.raw(), self.raw(self.points + (100, 0))], 10 + i / 30, ids=[2, 1])
        for player in players:
            plain = player_from_pose(player.to_dict((640, 480)))
            mirrored = player_from_pose(player.to_dict((640, 480), mirror=True))
            self.assertEqual(plain.id, player.id)
            for a, b in zip(plain.pose, mirrored.pose):
                self.assertAlmostEqual(b.position[0], 1 - a.position[0], places=4)
                self.assertEqual(b.position[1], a.position[1])
                self.assertEqual(b.state, JointState.TRACKED)
        payload = encode_frame([player_from_pose(p.to_dict((640, 480), mirror=True)) for p in players],
                               session_id="0" * 32, sequence=0, captured_time_s=10.14, sent_time_s=10.15, clock="local")
        self.assertEqual([p["id"] for p in json.loads(payload)["players"]], [1, 2])

    def test_prediction_then_loss_releases_reach(self):
        for i in range(5):
            players = self.tracker.update([self.raw()], 10 + i / 30, ids=[1])
        self.assertTrue(player_from_pose(players[0].to_dict((640, 480))).hand_raised[0])
        predicted = self.tracker.update([], 10.2, ids=[])[0]
        self.assertFalse(predicted.detected)
        p = player_from_pose(predicted.to_dict((640, 480)))
        self.assertTrue(p.tracked)
        self.assertEqual(p.pose[Joint.LEFT_WRIST].state, JointState.PREDICTED)
        self.assertTrue(p.hand_raised[0])
        lost = player_from_pose(self.tracker.update([], 10.6, ids=[])[0].to_dict((640, 480)))
        self.assertFalse(lost.tracked)
        self.assertEqual((lost.pose, lost.hand_raised, lost.events), ((), (), ()))

    def test_real_pull_event_keeps_its_id_across_snapshots(self):
        seen = []
        for i in range(46):
            points = self.points.copy()
            points[Joint.LEFT_WRIST, 1] += 240 * min(1, max(0, (i - 19) / 6))
            player = self.tracker.update([self.raw(points)], 10 + i / 30, ids=[1])[0]
            result = player_from_pose(player.to_dict((640, 480), mirror=True))
            seen.extend((i / 30, event) for event in result.events)
        self.assertGreaterEqual(len(seen), 5)
        self.assertEqual({event.id for _, event in seen}, {0})
        self.assertEqual({(event.type, event.side) for _, event in seen}, {("pull", "left")})
        self.assertLess(seen[-1][0] - seen[0][0], 0.2 + 1e-9)


if __name__ == "__main__":
    unittest.main()
