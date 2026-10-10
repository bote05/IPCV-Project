import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import types
import unittest
from unittest.mock import Mock, patch

from bridge.demo import demo_players
from bridge.face import encode_face, FACE_HEADER
from bridge.pipeline import TrackingResult, player_from_pose
from bridge.protocol import Joint, JointState, Landmark, MotionEvent, MotionSignal, PlayerState, encode_frame
from bridge.sender import UdpSender
import main

SESSION = "00000000000000000000000000000001"
JPEG = (Path(__file__).resolve().parents[1] / "demo_assets/player1.jpg").read_bytes()


class ProtocolTests(unittest.TestCase):
    def encode(self, players, **changes):
        fields = dict(session_id=SESSION, sequence=0, captured_time_s=1000.1, sent_time_s=1000.2, clock="qpc")
        fields.update(changes)
        return encode_frame(players, **fields)

    def test_complete_two_player_contract(self):
        frame = json.loads(self.encode(demo_players(0)))
        self.assertEqual([p["id"] for p in frame["players"]], [1, 2])
        self.assertEqual(len(frame["players"][0]["pose"]), 13)
        self.assertEqual(frame["players"][0]["pose"][0], {"position": [0.3, 0.2], "state": 3})
        self.assertEqual(len(frame["players"][0]["head_rotation_deg"]), 3)
        self.assertEqual(frame["players"][1]["world_position_m"], [0.4, 0, 2])
        self.assertEqual(frame["players"][0]["motion_signals"][0]["name"], "left_reach")

    def test_precise_processing_duration(self):
        frame = json.loads(self.encode([], captured_time_s=1000.123456, sent_time_s=1000.123789))
        self.assertAlmostEqual(frame["processing_ms"], 0.333, places=6)

    def test_explicit_loss_and_unknown_data_are_empty(self):
        self.assertEqual(json.loads(self.encode([]))["players"], [])
        p = json.loads(self.encode([PlayerState(1, False)]))["players"][0]
        for key in ("pose", "face_bbox", "head_rotation_deg", "world_position_m", "motion_signals", "events"):
            self.assertEqual(p[key], [])

    def test_invalid_payloads(self):
        for players in (
            [PlayerState(True, True)], [PlayerState(3, True)],
            [PlayerState(1, True), PlayerState(1, True)],
            [PlayerState(1, True)] * 3,
            [PlayerState(1, False, world_position_m=(0, 0, 2))],
            [PlayerState(1, True, (Landmark((0, 0), JointState.TRACKED),))],
            [PlayerState(1, True, (Landmark((0, 0), JointState.TRACKED),) * 33)],
            [PlayerState(1, True, face_bbox=(0, 0, -1, 1))],
            [PlayerState(1, True, world_position_m=(0, 2))],
            [PlayerState(1, True, head_rotation_deg=(0, float("nan"), 0))],
            [PlayerState(1, True, motion_signals=(MotionSignal("left_reach", 1),))],
            [PlayerState(1, True, motion_signals=(MotionSignal("a", True),) * 2)],
        ):
            with self.subTest(players=players), self.assertRaises(ValueError):
                self.encode(players)
        for fields in ({"sequence": -1}, {"session_id": "bad"}, {"clock": "unix"},
                       {"sent_time_s": 999}, {"captured_time_s": float("nan")}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.encode([], **fields)

    def test_landmark_states_and_nonfinite_coordinates(self):
        for position, state in (((float("inf"), 0), 3), ((0, 0, 0), 3), ((0, 0), -1),
                                ((0, 0), 4), ((0, 0), True), ((0, 0), 2.5)):
            with self.subTest(position=position), self.assertRaises(ValueError):
                self.encode([PlayerState(1, True, (Landmark(position, state),) * 13)])

    def test_event_contract_and_order(self):
        pull = MotionEvent(0, "pull", "left", 0.7, 1000.0)
        jump = MotionEvent(1, "jump", "", 2.1, 1000.1)
        frame = json.loads(self.encode([PlayerState(1, True, events=(pull, jump))]))
        self.assertEqual(frame["players"][0]["events"][0],
                         dict(id=0, type="pull", side="left", strength=0.7, time=1000.0))
        for events in ((pull, pull), (jump, pull), (MotionEvent(-1, "pull", "left", 1, 1),),
                       (MotionEvent(2**63, "jump", "", 1, 1),), (MotionEvent(0, "pull", "", 1, 1),),
                       (MotionEvent(0, "jump", "", float("nan"), 1),)):
            with self.subTest(events=events), self.assertRaises(ValueError):
                self.encode([PlayerState(1, True, events=events)])
        with self.assertRaises(ValueError):
            self.encode([PlayerState(1, False, events=(pull,))])

    def test_tracking_size_limit(self):
        self.assertLess(len(self.encode(demo_players(0))), 16384)
        with patch("bridge.protocol.MAX_PACKET_BYTES", 100), self.assertRaises(ValueError):
            self.encode(demo_players(0))

    def test_jpeg_protocol_and_size_limit(self):
        packet = encode_face(SESSION, 1, 7, JPEG)
        self.assertEqual(FACE_HEADER.unpack(packet[:45]), (b"IPCF", SESSION.encode(), 1, 7))
        self.assertEqual(packet[45:], JPEG)
        for jpeg in (b"not jpeg", b"\xff\xd8" + b"x" * 60000 + b"\xff\xd9"):
            with self.subTest(size=len(jpeg)), self.assertRaises(ValueError):
                encode_face(SESSION, 1, 0, jpeg)


class SenderTests(unittest.TestCase):
    def bind(self):
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(3)
        return receiver

    def test_real_udp_states_crops_restart_and_loss(self):
        with self.bind() as states, self.bind() as faces:
            port, face_port = states.getsockname()[1], faces.getsockname()[1]
            with UdpSender(port=port, face_port=face_port) as sender:
                sender.send(demo_players(0), captured_time_s=time.perf_counter() - 0.002)
                first = json.loads(states.recv(16384))
                self.assertGreater(first["processing_ms"], 1)
                sender.send_face(1, JPEG)
                crop = faces.recv(60000)
                self.assertEqual(FACE_HEADER.unpack(crop[:45])[1].decode(), first["session_id"])
                sender.send([PlayerState(1, False)])
                lost = json.loads(states.recv(16384))
                self.assertEqual(lost["sequence"], 1)
                self.assertFalse(lost["players"][0]["tracked"])
            with UdpSender(port=port, face_port=face_port) as restarted:
                restarted.send([])
                new = json.loads(states.recv(16384))
                self.assertEqual(new["sequence"], 0)
                self.assertNotEqual(first["session_id"], new["session_id"])

    def test_demo_cli_sends_both_channels_then_empty_state(self):
        with self.bind() as states, self.bind() as faces:
            result = subprocess.run([
                sys.executable, str(Path(main.__file__)), "--demo", "--duration", "0.15",
                "--port", str(states.getsockname()[1]), "--face-port", str(faces.getsockname()[1]),
            ], capture_output=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            frames = []
            while not frames or frames[-1]["players"]:
                frames.append(json.loads(states.recv(16384)))
            self.assertEqual([f["sequence"] for f in frames], list(range(len(frames))))
            self.assertGreater(len(frames), 3)
            self.assertEqual(len(frames[0]["players"]), 2)
            self.assertTrue(faces.recv(60000).startswith(b"IPCF"))

    def test_buffer_pressure_and_bad_payload(self):
        with UdpSender() as sender:
            with patch.object(sender, "_socket") as fake:
                fake.sendto.side_effect = BlockingIOError
                self.assertFalse(sender.send([]))
            self.assertEqual(sender.frames_dropped, 1)
            self.assertEqual(sender.sequence, 1)
            with self.assertRaises(ValueError):
                sender.send([PlayerState(7, True)])
            self.assertEqual(sender.sequence, 1)

    def test_destination_validation(self):
        for args in ({"host": "192.168.1.2"}, {"port": 0}, {"port": 5006}, {"face_port": 65536}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                UdpSender(**args)


class CameraRunnerTests(unittest.TestCase):
    def fake_cv(self):
        frame = object()
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.return_value = (True, frame)
        cv = types.SimpleNamespace(VideoCapture=Mock(return_value=camera),
                                   flip=Mock(return_value="mirrored preview"), imshow=Mock(),
                                   waitKey=Mock(side_effect=[0, ord("q")]), destroyAllWindows=Mock())
        return cv, camera, frame

    def test_canonical_unmirrored_frame_and_processor_failure_recovery(self):
        cv, camera, frame = self.fake_cv()
        processor = Mock(side_effect=[RuntimeError("module unavailable"), TrackingResult(demo_players(0), {1: JPEG})])
        sender = Mock()
        with patch.dict(sys.modules, {"cv2": cv}), self.assertLogs(level="INFO"):
            main.run_camera(sender, processor, 0, 240)
        for index, call in enumerate(processor.call_args_list):
            self.assertIs(call.args[0], frame)
            self.assertGreater(call.args[1], 0)
            if index == 1:
                self.assertEqual(call.args[1], sender.send.call_args_list[1].kwargs["captured_time_s"])
        self.assertEqual(sender.send.call_args_list[0].args, ((),))
        self.assertEqual(len(sender.send.call_args_list[1].args[0]), 2)
        cv.flip.assert_called_with(frame, 1)
        camera.release.assert_called_once()
        cv.destroyAllWindows.assert_called_once()

    def test_rejected_face_crop_keeps_valid_tracking(self):
        cv, camera, frame = self.fake_cv()
        processor = Mock(return_value=TrackingResult(demo_players(0), {1: b"not a jpeg", 2: JPEG}))
        with UdpSender() as real, patch.dict(sys.modules, {"cv2": cv}), self.assertLogs(level="WARNING"):
            sender = Mock(wraps=real)
            main.run_camera(sender, processor, 0, 240)
        self.assertEqual([len(call.args[0]) for call in sender.send.call_args_list], [2, 2])
        self.assertEqual([call.args[0] for call in sender.send_face.call_args_list], [1, 2])

    def test_camera_failure_sends_loss_and_releases_resources(self):
        cv, camera, frame = self.fake_cv()
        camera.read.return_value = (False, None)
        sender = Mock()
        with patch.dict(sys.modules, {"cv2": cv}), self.assertRaisesRegex(RuntimeError, "stopped"):
            main.run_camera(sender, Mock(), 0, 30)
        sender.send.assert_called_once_with(())
        camera.release.assert_called_once()


class PoseAdapterTests(unittest.TestCase):
    def data(self):
        return dict(id=1, valid=True, detected=False, hand_raised=[True, False],
                    joints=[dict(name=j.name.lower(), state=2, x=0.25, y=0.75) for j in Joint],
                    events=[dict(id=0, type="pull", side="left", strength=0.8, time=10.0)])

    def test_predicted_pose_and_repeated_events_are_preserved(self):
        player = player_from_pose(self.data())
        self.assertTrue(player.tracked)
        self.assertEqual(player.pose[Joint.LEFT_WRIST], Landmark((0.25, 0.75), JointState.PREDICTED))
        self.assertEqual(player.motion_signals, (MotionSignal("left_reach", True), MotionSignal("right_reach", False)))
        self.assertEqual(player.events, (MotionEvent(0, "pull", "left", 0.8, 10.0),))
        self.assertEqual(player.world_position_m, ())

    def test_invalid_body_clears_pose_actions_and_events(self):
        data = self.data()
        data["valid"] = False
        self.assertEqual(player_from_pose(data), PlayerState(1, False))

    def test_automatic_track_ids_and_wrong_joint_order_are_rejected(self):
        for player_id in (0, 3, True):
            data = self.data()
            data["id"] = player_id
            with self.subTest(player_id=player_id), self.assertRaises(ValueError):
                player_from_pose(data)
        data = self.data()
        data["joints"].reverse()
        with self.assertRaises(ValueError):
            player_from_pose(data)


if __name__ == "__main__":
    unittest.main()
