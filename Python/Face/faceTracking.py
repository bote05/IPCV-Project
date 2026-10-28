"""Task 1: face detection, smoothing and 256x256 face crops.

    face = FaceTracker()
    for frame, t in camera:                 # BGR frame, capture time in seconds
        faces = face.process(frame, t)      # list[FaceObservation]
    face.close()

Every visible face is returned, smoothed with a One Euro filter per face. A face that stops being
detected is still returned (detected=False, last box and JPEG) for FACE_TIMEOUT_S, then dropped.
This module does not decide who is player 1 or 2: it returns template-match scores as a hint
for Identity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from bridge.face import MAX_FACE_JPEG_BYTES
from Pose.filtering import OneEuroFilter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).resolve().parents[2] / 'Models'
MODEL_PATH = MODELS_DIR / 'blaze_face_short_range.tflite'   # python -m Face.download_model

FACE_JPEG_SIZE = (256, 256)    # what Unity expects (see bridge/README.md)
JPEG_QUALITIES = range(85, 9, -5)   # tried in order until the JPEG fits MAX_FACE_JPEG_BYTES

FACE_TIMEOUT_S = 0.5           # a face is dropped after this long without a detection
MIN_MATCH_IOU = 0.3            # min box overlap to treat a detection as the same face

# One Euro filter on (centre x, centre y, width, height) in pixels.
# Speed is measured in face widths per second, so the filter behaves the same at any distance.
FILTER_MIN_CUTOFF = 1.0        # Hz at rest; lower removes more jitter
FILTER_BETA = 1.0              # Hz added per face width/s; higher reduces lag during motion
FILTER_D_CUTOFF = 1.0          # Hz, smoothing of the speed estimate
MIN_DT = 1e-3

NUM_TEMPLATES = 2
TEMPLATE_SIZE = (64, 64)       # faces are resized to this before comparing

# Debug drawing (only used when FaceTracker(debug=True))
MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1.2
FONT_THICKNESS = 2
DETECTED_COLOR = (0, 200, 0)   # green (BGR)
HELD_COLOR = (160, 160, 160)   # grey
KEYPOINT_COLOR = (0, 255, 255)

_ROW = np.array([True])        # OneEuroFilter works on rows of an array; a face is one row


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FaceObservation:
    """One face in one frame.

    bbox: (x, y, width, height) normalized to the image size, origin top left, NOT mirrored.
          Smoothed, and clipped to the image.
    jpeg: 256x256 face crop, at most MAX_FACE_JPEG_BYTES; None if it could not be made small enough.
    template_scores: similarity (-1..1) to template 0 and template 1, or None until both
          templates exist. Template 0 is the face that was on the left of the camera image when
          the templates were captured, template 1 the one on the right. Higher = more similar.
    detected: True if the detector saw this face in this frame; False while it is being held
          after a missed detection (box, JPEG and scores are then from the last detection).
    track_id: stays the same for the same face across frames; a new id after the face is dropped.
          This is not a player id.
    """

    track_id: int
    bbox: Tuple[float, float, float, float]
    jpeg: Optional[bytes]
    template_scores: Optional[Tuple[float, float]]
    detected: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def model_missing_message() -> str:
    return (f'Face model not found at {MODEL_PATH}. '
            'Download it from the Python folder with: python -m Face.download_model')


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Overlap of two (cx, cy, w, h) boxes, 0..1."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = w * h
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def clip_box(box: np.ndarray, width: int, height: int) -> Tuple[int, int, int, int]:
    """(cx, cy, w, h) in pixels -> integer (x1, y1, x2, y2) inside the image."""
    x1 = int(round(max(box[0] - box[2] / 2, 0)))
    y1 = int(round(max(box[1] - box[3] / 2, 0)))
    x2 = int(round(min(box[0] + box[2] / 2, width)))
    y2 = int(round(min(box[1] + box[3] / 2, height)))
    return x1, y1, x2, y2


def make_template(face_crop: np.ndarray) -> np.ndarray:
    """Normalizes a face crop so two crops can be compared fairly."""
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, TEMPLATE_SIZE)
    return cv2.equalizeHist(gray)   # reduces the effect of lighting changes


def similarity(template_a: np.ndarray, template_b: np.ndarray) -> float:
    """Normalized correlation between two same-size templates (-1..1)."""
    return float(cv2.matchTemplate(template_a, template_b, cv2.TM_CCOEFF_NORMED)[0][0])


def encode_jpeg(image: np.ndarray) -> Optional[bytes]:
    """Encodes as JPEG, lowering quality until it fits one face packet. None if it never fits."""
    for quality in JPEG_QUALITIES:
        ok, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok and buf.size <= MAX_FACE_JPEG_BYTES:
            return buf.tobytes()
    return None


def _normalized_to_pixel(x: float, y: float, width: int, height: int) -> Optional[Tuple[int, int]]:
    if not (0 <= x <= 1 and 0 <= y <= 1):
        return None
    return min(math.floor(x * width), width - 1), min(math.floor(y * height), height - 1)


# ---------------------------------------------------------------------------
# Face tracks
# ---------------------------------------------------------------------------
@dataclass
class _Track:
    track_id: int
    filter: OneEuroFilter
    last_update_s: float
    last_detected_s: float
    jpeg: Optional[bytes] = None
    template: Optional[np.ndarray] = None
    scores: Optional[Tuple[float, float]] = None
    detected: bool = True
    keypoints: list = field(default_factory=list)

    @property
    def box(self) -> np.ndarray:
        """Smoothed (cx, cy, w, h) in pixels."""
        return self.filter.value[0]


class FaceTracker:
    """Keeps the detector, face tracks and templates alive between frames."""

    def __init__(self, debug: bool = False) -> None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(model_missing_message())
        options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO)
        self._detector = vision.FaceDetector.create_from_options(options)
        self.debug = debug
        self.templates: Optional[List[np.ndarray]] = None
        self._tracks: List[_Track] = []
        self._ids = count(1)
        self._last_timestamp_ms = -1

    # -- public -------------------------------------------------------------
    def process(self, frame_bgr: np.ndarray, t: float) -> List[FaceObservation]:
        """Detects, smooths and crops the faces in one frame.

        frame_bgr: normal (unmirrored) camera image. t: capture time in seconds, increasing.
        """
        height, width = frame_bgr.shape[:2]
        detections = self._detect(frame_bgr, t)
        self._update_tracks(detections, t)

        for track in self._tracks:
            if track.detected:
                self._refresh_crop(track, frame_bgr)
        self._capture_templates()
        self._score()

        if self.debug:
            cv2.imshow('Face tracking', self._draw(frame_bgr))

        return [self._observation(track, width, height) for track in self._tracks]

    def reset_templates(self) -> None:
        """Forget both templates; new ones are captured the next time exactly two faces are seen."""
        self.templates = None
        for track in self._tracks:
            track.scores = None

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> 'FaceTracker':
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- steps ----------------------------------------------------------------
    def _detect(self, frame_bgr: np.ndarray, t: float) -> List[Tuple[np.ndarray, list]]:
        """Raw detections as ((cx, cy, w, h) in pixels, keypoints)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(int(t * 1000), self._last_timestamp_ms + 1)   # must strictly increase
        self._last_timestamp_ms = timestamp_ms
        result = self._detector.detect_for_video(image, timestamp_ms)

        detections = []
        for detection in result.detections:
            b = detection.bounding_box
            if b.width > 0 and b.height > 0:
                box = np.array([b.origin_x + b.width / 2, b.origin_y + b.height / 2,
                                b.width, b.height], dtype=float)
                detections.append((box, list(detection.keypoints)))
        return detections

    def _update_tracks(self, detections: List[Tuple[np.ndarray, list]], t: float) -> None:
        """Matches detections to tracks by overlap, filters matched boxes, starts and drops tracks."""
        pairs = sorted(((iou(box, track.box), d, k)
                        for d, (box, _) in enumerate(detections)
                        for k, track in enumerate(self._tracks)), reverse=True)
        matched_detections, matched_tracks = set(), set()
        for overlap, d, k in pairs:
            if overlap < MIN_MATCH_IOU:
                break
            if d in matched_detections or k in matched_tracks:
                continue
            matched_detections.add(d)
            matched_tracks.add(k)
            track, (box, keypoints) = self._tracks[k], detections[d]
            dt = max(t - track.last_update_s, MIN_DT)
            track.filter.update(box[None, :], dt, _ROW, scale=max(track.box[2], 1.0))
            track.last_update_s = track.last_detected_s = t
            track.detected, track.keypoints = True, keypoints

        for k, track in enumerate(self._tracks):
            if k not in matched_tracks:
                track.detected = False
        self._tracks = [track for track in self._tracks
                        if track.detected or t - track.last_detected_s <= FACE_TIMEOUT_S]

        for d, (box, keypoints) in enumerate(detections):
            if d not in matched_detections:
                euro = OneEuroFilter((1, 4), FILTER_MIN_CUTOFF, FILTER_BETA, FILTER_D_CUTOFF)
                euro.set(_ROW, box)
                self._tracks.append(_Track(next(self._ids), euro, t, t, keypoints=keypoints))

    def _refresh_crop(self, track: _Track, frame_bgr: np.ndarray) -> None:
        """New JPEG and template from the smoothed box. A crop that is too big is skipped (None)."""
        height, width = frame_bgr.shape[:2]
        x1, y1, x2, y2 = clip_box(track.box, width, height)
        if x2 <= x1 or y2 <= y1:
            return
        crop = frame_bgr[y1:y2, x1:x2]
        track.jpeg = encode_jpeg(cv2.resize(crop, FACE_JPEG_SIZE))
        track.template = make_template(crop)

    def _capture_templates(self) -> None:
        """The first time exactly two faces are detected, store them left to right."""
        if self.templates is not None:
            return
        visible = [track for track in self._tracks if track.detected and track.template is not None]
        if len(visible) == NUM_TEMPLATES:
            visible.sort(key=lambda track: track.box[0])
            self.templates = [track.template for track in visible]

    def _score(self) -> None:
        if self.templates is None:
            return
        for track in self._tracks:
            if track.detected and track.template is not None:
                track.scores = tuple(similarity(track.template, template)
                                     for template in self.templates)

    def _observation(self, track: _Track, width: int, height: int) -> FaceObservation:
        x1, y1, x2, y2 = clip_box(track.box, width, height)
        bbox = (x1 / width, y1 / height, max(x2 - x1, 0) / width, max(y2 - y1, 0) / height)
        return FaceObservation(track.track_id, bbox, track.jpeg, track.scores, track.detected)

    def _draw(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Debug view: only the face regions on black, with track id and template scores."""
        height, width = frame_bgr.shape[:2]
        canvas = np.zeros_like(frame_bgr)
        for track in self._tracks:
            x1, y1, x2, y2 = clip_box(track.box, width, height)
            color = DETECTED_COLOR if track.detected else HELD_COLOR
            canvas[y1:y2, x1:x2] = frame_bgr[y1:y2, x1:x2]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            for keypoint in track.keypoints:
                point = _normalized_to_pixel(keypoint.x, keypoint.y, width, height)
                if point is not None:
                    cv2.circle(canvas, point, 2, KEYPOINT_COLOR, 2)
            label = f'#{track.track_id}'
            if track.scores is not None:
                label += f' {track.scores[0]:.2f}/{track.scores[1]:.2f}'
            cv2.putText(canvas, label, (x1 + MARGIN, y1 + MARGIN + ROW_SIZE),
                        cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, color, FONT_THICKNESS)
        return canvas
