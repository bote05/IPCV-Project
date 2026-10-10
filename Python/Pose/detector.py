"""Multi-person keypoint estimation with the MediaPipe Pose Landmarker (BlazePose GHUM)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

from .motion import torso_length
from .skeleton import MEDIAPIPE_INDEX, NUM_JOINTS

MODELS_DIR = Path(__file__).resolve().parents[2] / "Models"
MODEL_VARIANTS = ("lite", "full", "heavy")


def model_path(variant: str) -> Path:
    return MODELS_DIR / f"pose_landmarker_{variant}.task"


@dataclass(frozen=True)
class RawPose:
    """Unfiltered joints of one person in one frame.

    points: (NUM_JOINTS, 2) pixel coordinates.
    confidence: (NUM_JOINTS,) min(visibility, presence); low when a joint is occluded or out of frame.
    """

    points: np.ndarray
    confidence: np.ndarray

    def hide(self, joints) -> RawPose:
        """Copy with `joints` reported as undetected, to simulate occlusion."""
        confidence = self.confidence.copy()
        confidence[joints] = 0.0
        return RawPose(self.points, confidence)


class PoseDetector:
    """Runs the landmarker in VIDEO mode, which tracks people between frames instead of
    re-running the person detector every frame, and returns the selected joints per person.

    Detections whose torso is shorter than `min_torso` of the frame height are dropped: the
    landmarker occasionally fits a collapsed skeleton onto background clutter, with high joint
    confidence, while a player in front of a webcam has a torso of at least ~15% of the frame.
    """

    def __init__(
        self,
        model: str | Path = "full",
        num_poses: int = 2,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_torso: float = 0.05,
    ):
        path = model_path(model) if model in MODEL_VARIANTS else Path(model)
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found, run `python -m Pose.download_models` first")
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1
        self._min_torso = min_torso

    def detect(self, frame_bgr: np.ndarray, t: float) -> list[RawPose]:
        """Detect everyone in `frame_bgr`; `t` is the capture time in seconds."""
        # VIDEO mode rejects timestamps that do not strictly increase.
        timestamp_ms = max(round(t * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
        height, width = frame_bgr.shape[:2]
        poses = [_to_raw_pose(landmarks, width, height) for landmarks in result.pose_landmarks]
        everything = np.ones(NUM_JOINTS, dtype=bool)
        return [pose for pose in poses if torso_length(pose.points, everything) >= self._min_torso * height]

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> PoseDetector:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _to_raw_pose(landmarks, width: int, height: int) -> RawPose:
    selected = [landmarks[i] for i in MEDIAPIPE_INDEX]
    points = np.array([(lm.x * width, lm.y * height) for lm in selected])
    confidence = np.array([min(lm.visibility or 0.0, lm.presence or 0.0) for lm in selected])
    return RawPose(points, confidence)
