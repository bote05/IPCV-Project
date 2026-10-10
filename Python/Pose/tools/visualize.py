"""OpenCV drawing of raw and filtered poses, joint states, motion events and a live signal plot."""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from ..detector import RawPose
from ..evaluation.metrics import contiguous_runs
from ..filtering import JointState
from ..skeleton import BONES, HIPS, NUM_JOINTS, SIDES, WRISTS, Joint
from ..tracker import PlayerPose

PLAYER_COLORS = ((255, 160, 40), (60, 200, 255), (120, 255, 120), (255, 90, 200))
STATE_COLORS = {
    JointState.TRACKED: (80, 220, 80),
    JointState.PREDICTED: (0, 165, 255),
    JointState.ACQUIRING: (255, 200, 0),
}
RAW_COLOR = (170, 170, 170)
EVENT_DISPLAY_TIME = 0.6


def player_color(player_id: int) -> tuple[int, int, int]:
    return PLAYER_COLORS[player_id % len(PLAYER_COLORS)]


def _pt(point: np.ndarray) -> tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))


def draw_text(image: np.ndarray, text: str, origin: tuple[int, int], color=(255, 255, 255), scale: float = 0.5) -> None:
    # A shadow rather than a thick outline: OpenCV 5 spaces glyphs differently per thickness.
    x, y = origin
    cv2.putText(image, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_raw(image: np.ndarray, pose: RawPose, min_confidence: float) -> None:
    visible = pose.confidence >= min_confidence
    for a, b in BONES:
        if visible[a] and visible[b]:
            cv2.line(image, _pt(pose.points[a]), _pt(pose.points[b]), RAW_COLOR, 1, cv2.LINE_AA)
    for j in np.flatnonzero(visible):
        cv2.drawMarker(image, _pt(pose.points[j]), RAW_COLOR, cv2.MARKER_CROSS, 7, 1)


def draw_player(image: np.ndarray, player: PlayerPose) -> None:
    color = player_color(player.id)
    shown = player.state >= JointState.PREDICTED
    if not shown.any():
        return
    for a, b in BONES:
        if shown[a] and shown[b]:
            solid = player.state[a] == JointState.TRACKED and player.state[b] == JointState.TRACKED
            cv2.line(image, _pt(player.position[a]), _pt(player.position[b]), color, 3 if solid else 1, cv2.LINE_AA)
    for j in range(NUM_JOINTS):
        state = JointState(player.state[j])
        if state == JointState.LOST:
            continue
        filled = state == JointState.TRACKED
        cv2.circle(image, _pt(player.position[j]), 5, STATE_COLORS[state], -1 if filled else 2, cv2.LINE_AA)

    for i, wrist in enumerate(WRISTS):
        if player.motion.hand_raised[i] and shown[wrist]:
            cv2.circle(image, _pt(player.position[wrist]), 16, color, 2, cv2.LINE_AA)

    head = player.position[Joint.NOSE] if shown[Joint.NOSE] else player.motion.center
    status = "" if player.detected else " (not detected)"
    draw_text(image, f"P{player.id}{status}", (int(head[0]) - 20, int(head[1]) - 30), color, 0.6)


class EventFlash:
    """Shows recent motion events next to the joint that produced them."""

    def __init__(self):
        self._events: list[tuple[float, PlayerPose, str, int]] = []

    def add(self, player: PlayerPose) -> None:
        for event in player.motion.events:
            joint = WRISTS[SIDES.index(event.side)] if event.side else HIPS[0]
            self._events.append((event.time, player, f"{event.kind.upper()} {event.side or ''}".strip(), joint))

    def draw(self, image: np.ndarray, now: float) -> None:
        self._events = [e for e in self._events if now - e[0] < EVENT_DISPLAY_TIME]
        for _, player, label, joint in self._events:
            x, y = _pt(player.position[joint])
            draw_text(image, label, (x + 12, y), player_color(player.id), 0.8)


class SignalPlot:
    """Scrolling plot of one coordinate: raw detections against the filtered estimate."""

    def __init__(self, title: str, window: float = 4.0):
        self.title = title
        self.window = window
        self._samples: deque[tuple[float, float, float, int]] = deque()

    def add(self, t: float, raw: float, filtered: float, state: int) -> None:
        self._samples.append((t, raw, filtered, state))
        while self._samples and t - self._samples[0][0] > self.window:
            self._samples.popleft()

    def draw(self, image: np.ndarray, rect: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
        """Draw into `rect` (x, y, width, height); orange bands mark frames where the joint was predicted."""
        x0, y0, width, height = rect
        region = image[y0 : y0 + height, x0 : x0 + width]
        region[:] = region // 3
        draw_text(image, f"{self.title}: raw (grey) / filtered", (x0 + 6, y0 + 16), scale=0.45)
        if len(self._samples) < 2:
            return
        times, raw, filtered, state = np.array(self._samples).T
        shown = state >= JointState.PREDICTED
        finite = np.isfinite(raw)
        values = np.concatenate([raw[finite], filtered[shown]])
        if values.size == 0:
            return
        low, span = values.min(), max(np.ptp(values), 1e-6)
        top, bottom = y0 + 24, y0 + height - 8
        xs = (x0 + (times - times[-1] + self.window) / self.window * width).astype(int)

        def ys(v: np.ndarray) -> np.ndarray:
            return (top + (v - low) / span * (bottom - top)).astype(int)

        for x in xs[state == JointState.PREDICTED]:
            cv2.line(image, (x, top), (x, bottom), STATE_COLORS[JointState.PREDICTED], 1)
        for x, y in zip(xs[finite], ys(raw[finite])):
            cv2.circle(image, (x, y), 2, RAW_COLOR, -1)
        segments = [np.stack([xs[run], ys(filtered[run])], axis=1).astype(np.int32) for run in contiguous_runs(shown, 2)]
        if segments:
            cv2.polylines(image, segments, False, color, 2, cv2.LINE_AA)
