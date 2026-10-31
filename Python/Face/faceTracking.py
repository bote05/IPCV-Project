"""Task 1: face detection, smoothing and 256x256 face crops.

    face = FaceTracker()
    for frame, t in camera:                 # BGR frame, capture time in seconds
        faces = face.process(frame, t)      # list[FaceObservation]
    face.close()

Every visible face is returned, smoothed with a One Euro filter per face. A face that stops being
detected is still returned (detected=False, last box and JPEG) for FACE_TIMEOUT_S, then dropped.
This module does not decide who is player 1 or 2: it returns face-recognition scores against the
two stored player faces as a hint for Identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from bridge.face import MAX_FACE_JPEG_BYTES
from Pose.filtering import OneEuroFilter

MODELS_DIR = Path(__file__).resolve().parents[2] / 'Models'
MODEL_PATH = MODELS_DIR / 'face_detection_yunet_2023mar.onnx'   # python -m Face.download_model
RECOGNIZER_PATH = MODELS_DIR / 'face_recognition_sface_2021dec.onnx'

DETECT_WIDTH = 640             # frames are downscaled to this width before detection
MIN_SCORE = 0.7                # detector confidence needed to accept a face

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

# Debug drawing (only used when FaceTracker(debug=True))
MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1.2
FONT_THICKNESS = 2
DETECTED_COLOR = (0, 200, 0)   # green (BGR)
HELD_COLOR = (160, 160, 160)   # grey
KEYPOINT_COLOR = (0, 255, 255)

_ROW = np.array([True])        # OneEuroFilter works on rows of an array; a face is one row


@dataclass(frozen=True)
class FaceObservation:
    """One face in one frame.

    bbox: (x, y, width, height) normalized to the image size, origin top left, NOT mirrored.
          Smoothed, and clipped to the image.
    jpeg: 256x256 face crop, at most MAX_FACE_JPEG_BYTES; None if it could not be made small enough.
    template_scores: cosine similarity (-1..1) to the two stored SFace features, or None until both
          templates exist. Template 0 is the face that was on the left of the mirrored screen
          (the right of the camera image) when the templates were captured, template 1 the other
          one. Higher = more similar.
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


def model_missing_message() -> str:
    return (f'Face models not found at {MODEL_PATH} and {RECOGNIZER_PATH}. '
            'Download them from the Python folder with: python -m Face.download_model')


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
    x1 = int(round(np.clip(box[0] - box[2] / 2, 0, width)))
    y1 = int(round(np.clip(box[1] - box[3] / 2, 0, height)))
    x2 = int(round(np.clip(box[0] + box[2] / 2, 0, width)))
    y2 = int(round(np.clip(box[1] + box[3] / 2, 0, height)))
    return x1, y1, x2, y2


def encode_jpeg(image: np.ndarray) -> Optional[bytes]:
    """Encodes as JPEG, lowering quality until it fits one face packet. None if it never fits."""
    for quality in JPEG_QUALITIES:
        ok, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok and buf.size <= MAX_FACE_JPEG_BYTES:
            return buf.tobytes()
    return None


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
        if not MODEL_PATH.is_file() or not RECOGNIZER_PATH.is_file():
            raise FileNotFoundError(model_missing_message())
        self._detector = cv2.FaceDetectorYN.create(str(MODEL_PATH), '', (DETECT_WIDTH, DETECT_WIDTH), MIN_SCORE)
        self._recognizer = cv2.FaceRecognizerSF.create(str(RECOGNIZER_PATH), '')
        self.debug = debug
        self.templates: Optional[List[np.ndarray]] = None
        self._tracks: List[_Track] = []
        self._ids = count(1)

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
        self._tracks.clear()

    def close(self) -> None:
        self._detector = None

    def __enter__(self) -> 'FaceTracker':
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- steps ----------------------------------------------------------------
    def _detect(self, frame_bgr: np.ndarray, t: float) -> List[Tuple[np.ndarray, list]]:
        """Raw detections as ((cx, cy, w, h) in pixels, keypoints in pixels)."""
        scale = min(1.0, DETECT_WIDTH / frame_bgr.shape[1])
        small = cv2.resize(frame_bgr, None, fx=scale, fy=scale) if scale < 1 else frame_bgr
        self._detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self._detector.detect(small)

        detections = []
        for face in faces if faces is not None else []:
            x, y, w, h = face[:4] / scale
            if w > 0 and h > 0:
                box = np.array([x + w / 2, y + h / 2, w, h], dtype=float)
                keypoints = list(face[4:14].reshape(5, 2) / scale)   # eyes, nose tip, mouth corners
                detections.append((box, keypoints))
        return detections

    def _update_tracks(self, detections: List[Tuple[np.ndarray, list]], t: float) -> None:
        """Matches detections to tracks by overlap, filters matched boxes, starts and drops tracks."""
        self._tracks = [track for track in self._tracks if t - track.last_detected_s <= FACE_TIMEOUT_S]
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

        for d, (box, keypoints) in enumerate(detections):
            if d not in matched_detections:
                euro = OneEuroFilter((1, 4), FILTER_MIN_CUTOFF, FILTER_BETA, FILTER_D_CUTOFF)
                euro.set(_ROW, box)
                self._tracks.append(_Track(next(self._ids), euro, t, t, keypoints=keypoints))

    def _refresh_crop(self, track: _Track, frame_bgr: np.ndarray) -> None:
        """New JPEG and template from the smoothed box. A crop that is too big is skipped (None)."""
        height, width = frame_bgr.shape[:2]
        x1, y1, x2, y2 = clip_box(track.box, width, height)
        track.jpeg = track.template = track.scores = None
        if x2 <= x1 or y2 <= y1:
            return
        crop = frame_bgr[y1:y2, x1:x2]
        track.jpeg = encode_jpeg(cv2.resize(crop, FACE_JPEG_SIZE))
        # SFace aligns the face on its five landmarks (eyes, nose, mouth corners) first.
        face_row = np.array([x1, y1, x2 - x1, y2 - y1, *np.ravel(track.keypoints), 1.0], dtype=np.float32)
        track.template = self._recognizer.feature(self._recognizer.alignCrop(frame_bgr, face_row)).copy()

    def _capture_templates(self) -> None:
        """The first time exactly two faces are detected, store them left to right on the mirrored screen."""
        if self.templates is not None:
            return
        visible = [track for track in self._tracks if track.detected and track.template is not None]
        if len(visible) == NUM_TEMPLATES:
            visible.sort(key=lambda track: -track.box[0])
            self.templates = [track.template for track in visible]

    def _score(self) -> None:
        if self.templates is None:
            return
        for track in self._tracks:
            if track.detected and track.template is not None:
                track.scores = tuple(float(self._recognizer.match(track.template, template, cv2.FaceRecognizerSF_FR_COSINE))
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
                cv2.circle(canvas, (int(keypoint[0]), int(keypoint[1])), 2, KEYPOINT_COLOR, 2)
            label = f'#{track.track_id}'
            if track.scores is not None:
                label += f' {track.scores[0]:.2f}/{track.scores[1]:.2f}'
            cv2.putText(canvas, label, (x1 + MARGIN, y1 + MARGIN + ROW_SIZE),
                        cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, color, FONT_THICKNESS)
        return canvas
