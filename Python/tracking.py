"""Runs Face, Pose and Identity on every webcam frame and builds what the bridge sends to Unity."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import math
import logging

import cv2

from bridge.pipeline import TrackingResult, player_from_pose
from Face.faceTracking import FaceTracker
from Identity.identity import Identity
from Identity.position import PositionEstimator
from Pose import PoseDetector, PoseTracker, Joint, JointState
from Pose.tools.demo import mirror_for_display
from Pose.tools.visualize import draw_player, draw_text, player_color


class GameTracking:
    def __init__(self, calibration_distance: float = 2.0, horizontal_fov_deg: float = 65.0):
        self.faces = FaceTracker()
        try:
            self.detector = PoseDetector("full", num_poses=2)
        except Exception:
            self.faces.close()
            raise
        self.face_thread = ThreadPoolExecutor(1)
        self.tracker = PoseTracker()
        self.identity = Identity(maxPlayers=2)
        self.calibration_distance = calibration_distance
        self.fov = horizontal_fov_deg
        self.positions = None
        self.players = []   # PlayerPose list of the last frame, Identity matches against it
        self.face_of = {}   # player id -> FaceObservation
        self.calibrate_next = False

    def process(self, frame, t: float) -> TrackingResult:
        height, width = frame.shape[:2]
        if self.positions is None:
            # No camera calibration: the focal length comes from an assumed field of view.
            fx = width / 2 / math.tan(math.radians(self.fov) / 2)
            self.positions = PositionEstimator(fx, fx, width / 2, height / 2)
        self.identity.imageSize = (width, height)

        face_job = self.face_thread.submit(self.faces.process, frame, t)
        try:
            poses = self.detector.detect(frame, t)
        except Exception:
            # Finish this frame before reset or shutdown touches the face tracker.
            face_job.result()
            raise
        faces = face_job.result()
        previous = [p for p in self.players if t - p.time <= self.tracker.config.filter.max_gap]
        ids, face_of = self.identity.assign(poses, previous, faces)
        self.players = self.tracker.update(poses, t, ids)
        for pid, previous in self.face_of.items():
            if pid in ids and pid not in face_of:
                held = next((f for f in faces if not f.detected and f.track_id == previous.track_id), None)
                if held is not None:
                    face_of[pid] = held
        self.face_of = {pid: face for pid, face in face_of.items() if face is not None}

        torso = [Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER, Joint.LEFT_HIP, Joint.RIGHT_HIP]
        positioned = [p for p in self.players if p.detected and all(p.state[j] == JointState.TRACKED for j in torso)]
        if self.calibrate_next:
            if {p.id for p in positioned} == {1, 2}:
                self.positions.K.clear()
                self.positions.estimate_players(positioned, self.calibration_distance)
                logging.info("Both players calibrated at %g m", self.calibration_distance)
            else:
                logging.warning("Calibration skipped: both players' shoulders and hips must be visible. Press C again.")
        metres = self.positions.estimate_players(positioned)
        self.calibrate_next = False

        states = []
        for player in self.players:
            state = player_from_pose(player.to_dict((width, height), mirror=True))
            if state.tracked:
                face = self.face_of.get(player.id)
                if face is not None:
                    x, y, w, h = face.bbox
                    state = replace(state, face_bbox=(1 - x - w, y, w, h))
                if player.id in metres:
                    x, y, z = metres[player.id]
                    state = replace(state, world_position_m=(x, -y, z))
            states.append(state)
        jpegs = {pid: face.jpeg for pid, face in self.face_of.items() if face.jpeg}
        return TrackingResult(tuple(states), jpegs, (width, height))

    def on_key(self, key: int) -> None:
        if key == ord("c"):
            self.calibrate_next = True
        elif key == ord("n"):
            # New people in front of the camera: forget faces, identities and calibration.
            self.faces.reset_templates()
            self.identity = Identity(maxPlayers=2)
            self.tracker.reset()
            self.players = []
            self.face_of = {}
            self.calibrate_next = False
            if self.positions is not None:
                self.positions.K.clear()

    def draw(self, frame):
        """Mirrored webcam view with each player's skeleton, number and face box."""
        height, width = frame.shape[:2]
        view = cv2.flip(frame, 1)
        _, players = mirror_for_display([], self.players, width)
        for player in players:
            draw_player(view, player)
        for pid, face in self.face_of.items():
            x, y, w, h = face.bbox
            x = 1 - x - w
            cv2.rectangle(view, (int(x * width), int(y * height)),
                          (int((x + w) * width), int((y + h) * height)), player_color(pid), 2)
        if self.faces.templates is None:
            draw_text(view, "Both players: stand side by side and face the camera", (10, 25), scale=0.6)
        calibrated = self.positions is not None and self.positions.K
        draw_text(view, f"C calibrate at {self.calibration_distance:g} m{' (done)' if calibrated else ''}"
                  "   N new players   Q quit", (10, height - 12), scale=0.5)
        return view

    def close(self) -> None:
        self.face_thread.shutdown()
        self.faces.close()
        self.detector.close()
