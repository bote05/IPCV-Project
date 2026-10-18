import math
import time
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = 'detector.tflite'
CAMERA_INDEX = 0

NUM_PLAYERS = 2
TEMPLATE_SIZE = (64, 64)   # faces are resized to this before comparing
MATCH_THRESHOLD = 0.3      # min similarity (-1..1) to accept a match
PLAYER_VIEW_SIZE = (200, 200)   # (width, height) of each stored player image

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


def visualize(image: np.ndarray, detections,
              assignment: List[Optional[int]]) -> np.ndarray:
    """Shows only the detected faces on a black background, labelled by player."""
    height, width, _ = image.shape
    annotated = np.zeros_like(image)

    for detection, player in zip(detections, assignment):
        x1, y1, x2, y2 = clamp_bbox(detection.bounding_box, width, height)
        color = PLAYER_COLORS[player] if player is not None else UNKNOWN_COLOR

        # Copy the face in first, then draw on top of it
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
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open webcam {CAMERA_INDEX}')

    # Templates captured once, kept only for this run
    player_templates: Optional[List[np.ndarray]] = None
    # Current face image of each player (None when not visible this frame)
    players: List[Optional[np.ndarray]] = [None] * NUM_PLAYERS



    start = time.monotonic()
    with create_detector() as detector:
        while True:
            ok, frame = cap.read()          # frame is BGR
            if not ok:
                print('Failed to read frame from webcam.')
                break

            # MediaPipe expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            # Keep only detections that produce a usable crop
            detections, crops = [], []
            for detection in result.detections:
                crop = crop_face(frame, detection.bounding_box)
                if crop is not None:
                    detections.append(detection)
                    crops.append(crop)

            # Capture templates the first time exactly two faces are visible.
            # Sorted left to right: left face = player 0, right face = player 1.
            if player_templates is None and len(crops) == NUM_PLAYERS:
                order = sorted(range(len(crops)),
                               key=lambda i: detections[i].bounding_box.origin_x)
                player_templates = [make_template(crops[i]) for i in order]
                print('Player templates captured.')

            # Identify each face and update the players array
            players = [None] * NUM_PLAYERS
            if player_templates is not None:
                face_templates = [make_template(c) for c in crops]
                assignment = assign_players(face_templates, player_templates)

                for crop, player in zip(crops, assignment):
                    if player is not None:
                        players[player] = cv2.resize(crop, PLAYER_VIEW_SIZE)
            else:
                assignment = [None] * len(detections)

            # Display
            cv2.imshow('Face Detection', visualize(frame, detections, assignment))
            for i, face in enumerate(players):
                if face is not None:
                    cv2.imshow(f'Player {i + 1}', face)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):              # re-capture templates
                player_templates = None
                print('Templates cleared - waiting for two faces.')

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()