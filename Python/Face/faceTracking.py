"""Face detection + player identification, plugged into main.py as a frame processor.

Run from the repository root:
    python Python/main.py --webcam --processor Face.faceTracking:process_frame
"""

import atexit
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from bridge.pipeline import TrackingResult
from bridge.protocol import PlayerState

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = str(Path(__file__).with_name('detector.tflite'))  # works from any cwd

NUM_PLAYERS = 2
TEMPLATE_SIZE = (64, 64)       # faces are resized to this before comparing
MATCH_THRESHOLD = 0.3          # min similarity (-1..1) to accept a match
FACE_JPEG_SIZE = (256, 256)    # what Unity expects (see bridge/README.md)
JPEG_QUALITY = 85             # starting quality; lowered if the JPEG is too big

# Operating system this runs on: 'windows' or 'mac'.
# macOS can only send UDP packets up to 9216 bytes, so on 'mac' the face JPEGs
# are compressed harder to fit. Windows allows the bridge's full 60 KB.
PLATFORM = 'mac'
MAX_JPEG_BYTES = {
    'windows': 60_000 - 45,   # bridge limit minus the 45-byte face header
    'mac': 9_216 - 45,        # macOS UDP limit minus the 45-byte face header
}
if PLATFORM not in MAX_JPEG_BYTES:
    raise ValueError(f"PLATFORM must be 'windows' or 'mac', not {PLATFORM!r}")
SHOW_DEBUG_WINDOWS = True      # main.py's waitKey keeps these windows updating

MARGIN = 10          # pixels
ROW_SIZE = 10        # pixels
FONT_SIZE = 1.5
FONT_THICKNESS = 2
PLAYER_COLORS = [(255, 128, 0), (0, 0, 255)]   # player 1 blue, player 2 red (BGR)
UNKNOWN_COLOR = (200, 200, 200)                # grey
KEYPOINT_COLOR = (0, 255, 0)                   # green


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float,
        image_width: int, image_height: int) -> Optional[Tuple[int, int]]:
    """Converts a normalized (0..1) coordinate pair to pixel coordinates."""

    def is_valid(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and \
               (value < 1 or math.isclose(1, value))

    if not (is_valid(normalized_x) and is_valid(normalized_y)):
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def clamp_bbox(bbox, width: int, height: int) -> Tuple[int, int, int, int]:
    """Returns (x1, y1, x2, y2) clipped to the image so slicing never wraps."""
    x1 = max(bbox.origin_x, 0)
    y1 = max(bbox.origin_y, 0)
    x2 = min(bbox.origin_x + bbox.width, width)
    y2 = min(bbox.origin_y + bbox.height, height)
    return x1, y1, x2, y2


def crop_face(frame: np.ndarray, bbox) -> Optional[np.ndarray]:
    """Cuts the face out of the frame. Returns None if the box is empty."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox, width, height)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def make_template(face_crop: np.ndarray) -> np.ndarray:
    """Normalizes a face crop so two crops can be compared fairly."""
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, TEMPLATE_SIZE)
    return cv2.equalizeHist(gray)   # reduces the effect of lighting changes


def similarity(template_a: np.ndarray, template_b: np.ndarray) -> float:
    """Normalized correlation between two same-size templates (-1..1)."""
    return float(cv2.matchTemplate(template_a, template_b,
                                   cv2.TM_CCOEFF_NORMED)[0][0])


def assign_players(face_templates: List[np.ndarray],
                   player_templates: List[np.ndarray]) -> List[Optional[int]]:
    """Matches each detected face to a player index (or None).

    All face/player pairs are scored, then the best pairs are taken first,
    so two faces can never be assigned to the same player.
    """
    scores = []
    for f, face_t in enumerate(face_templates):
        for p, player_t in enumerate(player_templates):
            scores.append((similarity(face_t, player_t), f, p))
    scores.sort(reverse=True)

    assignment: List[Optional[int]] = [None] * len(face_templates)
    used_players = set()
    for score, f, p in scores:
        if score < MATCH_THRESHOLD:
            break
        if assignment[f] is None and p not in used_players:
            assignment[f] = p
            used_players.add(p)
    return assignment


def encode_jpeg(image: np.ndarray) -> bytes:
    """Encodes as JPEG, lowering quality until it fits in one UDP packet on PLATFORM."""
    max_bytes = MAX_JPEG_BYTES[PLATFORM]
    for quality in range(JPEG_QUALITY, 9, -10):
        ok, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError('JPEG encoding failed')
        if buf.size <= max_bytes:
            break
    return buf.tobytes()


def mirrored_normalized_bbox(x1: int, y1: int, x2: int, y2: int,
                             width: int, height: int) -> Tuple[float, float, float, float]:
    """Pixel box -> (x, y, w, h) normalized to the image, mirrored for Unity's display."""
    x, y = x1 / width, y1 / height
    w, h = (x2 - x1) / width, (y2 - y1) / height
    return (1 - x - w, y, w, h)


def visualize(image: np.ndarray, detections,
              assignment: List[Optional[int]]) -> np.ndarray:
    """Shows only the detected faces on a black background, labelled by player."""
    height, width, _ = image.shape
    annotated = np.zeros_like(image)

    for detection, player in zip(detections, assignment):
        x1, y1, x2, y2 = clamp_bbox(detection.bounding_box, width, height)
        color = PLAYER_COLORS[player] if player is not None else UNKNOWN_COLOR

        annotated[y1:y2, x1:x2] = image[y1:y2, x1:x2]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        for keypoint in detection.keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(
                keypoint.x, keypoint.y, width, height)
            if keypoint_px is not None:
                cv2.circle(annotated, keypoint_px, 2, KEYPOINT_COLOR, 2)

        label = f'Player {player + 1}' if player is not None else 'Unknown'
        cv2.putText(annotated, label, (x1 + MARGIN, y1 + MARGIN + ROW_SIZE),
                    cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, color, FONT_THICKNESS)

    return annotated


def create_detector() -> vision.FaceDetector:
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO)
    return vision.FaceDetector.create_from_options(options)


# ---------------------------------------------------------------------------
# Frame processor
# ---------------------------------------------------------------------------
class FaceTracker:
    """Keeps the detector and player templates alive between frames."""

    def __init__(self) -> None:
        self.detector = create_detector()
        self.player_templates: Optional[List[np.ndarray]] = None
        # Last detected face of each player (index 0 = player 1).
        # Kept when the player is not detected; None until first seen.
        self.players: List[Optional[np.ndarray]] = [None] * NUM_PLAYERS
        self._last_jpegs: List[Optional[bytes]] = [None] * NUM_PLAYERS
        self._last_bboxes: List[Optional[Tuple[float, float, float, float]]] = [None] * NUM_PLAYERS
        self._last_timestamp_ms = -1

    def reset_templates(self) -> None:
        self.player_templates = None
        self.players = [None] * NUM_PLAYERS
        self._last_jpegs = [None] * NUM_PLAYERS
        self._last_bboxes = [None] * NUM_PLAYERS

    def close(self) -> None:
        self.detector.close()

    def process(self, frame_bgr: np.ndarray, captured_time_s: float) -> TrackingResult:
        height, width = frame_bgr.shape[:2]

        # MediaPipe expects RGB and strictly increasing timestamps in ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(int(captured_time_s * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        # Keep only detections that produce a usable crop
        detections, crops = [], []
        for detection in result.detections:
            crop = crop_face(frame_bgr, detection.bounding_box)
            if crop is not None:
                detections.append(detection)
                crops.append(crop)

        # Capture templates the first time exactly two faces are visible.
        # Sorted left to right in the camera image: left = player 1, right = player 2.
        if self.player_templates is None and len(crops) == NUM_PLAYERS:
            order = sorted(range(len(crops)),
                           key=lambda i: detections[i].bounding_box.origin_x)
            self.player_templates = [make_template(crops[i]) for i in order]
            print('Player templates captured.')

        if self.player_templates is not None:
            assignment = assign_players([make_template(c) for c in crops],
                                        self.player_templates)
        else:
            assignment = [None] * len(detections)

        # Update the stored face of every player detected this frame
        for detection, crop, index in zip(detections, crops, assignment):
            if index is None:
                continue
            face = cv2.resize(crop, FACE_JPEG_SIZE)
            x1, y1, x2, y2 = clamp_bbox(detection.bounding_box, width, height)
            self.players[index] = face
            self._last_jpegs[index] = encode_jpeg(face)
            self._last_bboxes[index] = mirrored_normalized_bbox(x1, y1, x2, y2, width, height)

        # Send every player seen at least once: the current face if detected,
        # otherwise the last one. (main.py only sends faces for players listed
        # as tracked with a face_bbox, so the last bbox is sent along too.)
        states, face_jpegs = [], {}
        for index in range(NUM_PLAYERS):
            if self._last_jpegs[index] is None:
                continue
            player_id = index + 1                      # bridge uses IDs 1 and 2
            states.append(PlayerState(player_id, True, face_bbox=self._last_bboxes[index]))
            face_jpegs[player_id] = self._last_jpegs[index]

        if SHOW_DEBUG_WINDOWS:
            cv2.imshow('Face Detection', visualize(frame_bgr, detections, assignment))
            for i, face in enumerate(self.players):
                if face is not None:
                    cv2.imshow(f'Player {i + 1}', face)

        return TrackingResult(players=tuple(states), face_jpegs=face_jpegs)


_tracker: Optional[FaceTracker] = None


def process_frame(frame_bgr: np.ndarray, captured_time_s: float) -> TrackingResult:
    """Entry point for main.py: --processor Face.faceTracking:process_frame"""
    global _tracker
    if _tracker is None:
        _tracker = FaceTracker()
        atexit.register(_tracker.close)
    return _tracker.process(frame_bgr, captured_time_s)
