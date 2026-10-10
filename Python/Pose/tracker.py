"""Per-player pose tracking: keeps each person's filter state attached to the same person across
frames and turns their detections into smoothed joints and motion signals.

MediaPipe returns people in no guaranteed order, so detections are matched to existing tracks by
their distance to each track's predicted joints (Hungarian assignment). This short-term matching is
only what temporal filtering needs; mapping tracks to persistent players across crossings, long
occlusions and re-entry is the identity module's job, which can pass its own ids to `update`.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from itertools import count

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detector import RawPose
from .filtering import FilterConfig, JointState, KeypointFilter
from .motion import MotionAnalyzer, MotionConfig, MotionEvent, MotionSignals, torso_length
from .skeleton import JOINT_NAMES, NUM_JOINTS

FALLBACK_SCALE = 150.0  # px, torso length of a player ~2 m from a 720p webcam; used until measured


@dataclass(frozen=True)
class TrackerConfig:
    match_distance: float = 1.0  # torso lengths; mean joint distance beyond which a detection is a new person
    duplicate_distance: float = 0.2  # torso lengths; detections closer than this are the same person
    track_timeout: float = 1.0  # s without a detection before a track is dropped
    event_hold: float = 0.2  # s an event stays in the output, so one lost message cannot lose it
    filter: FilterConfig = field(default_factory=FilterConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)


@dataclass(frozen=True)
class PlayerPose:
    id: int
    time: float
    detected: bool  # a detection was assigned to this player in the current frame
    position: np.ndarray  # (N, 2) filtered joints in pixels
    velocity: np.ndarray  # (N, 2) px/s
    state: np.ndarray  # (N,) JointState
    raw: RawPose | None
    motion: MotionSignals  # motion.events: the events fired in this frame
    events: tuple[MotionEvent, ...]  # events fired within the last `event_hold` seconds

    def to_dict(self, image_size: tuple[int, int], mirror: bool = False) -> dict:
        """JSON-ready summary for the game. Image coordinates are normalised to [0, 1] (origin top
        left); body-frame values are in torso lengths with y up. Lists of objects instead of maps
        keep it parseable by Unity's JsonUtility.

        Each event is repeated in every message for `event_hold` seconds; the game acts on each
        event id once. `mirror` flips everything horizontally for a mirrored display; joint names
        stay the player's own left and right, which then appear on the same side of the screen.
        """
        width, height = image_size
        m = self.motion
        flip = np.array([-1.0, 1.0]) if mirror else np.ones(2)

        def image_x(x: float) -> float:
            return 1.0 - x / width if mirror else x / width

        joints = [
            {
                "name": name,
                "state": int(self.state[j]),
                "x": _round(image_x(self.position[j, 0])),
                "y": _round(self.position[j, 1] / height),
                "body": _round(m.body[j] * flip),
                "velocity": _round(m.body_velocity[j] * flip),
                "moving": bool(m.moving[j]),
            }
            for j, name in enumerate(JOINT_NAMES)
        ]
        return {
            "id": self.id,
            "time": _round(self.time),
            "detected": self.detected,
            "valid": m.valid,
            "scale": _round(m.scale / height),
            "center": [_round(image_x(m.center[0])), _round(m.center[1] / height)],
            "center_velocity": _round(m.center_velocity * flip),
            "lean": _round(-m.lean if mirror else m.lean),
            "arm_extension": _round(m.arm_extension),
            "hand_raised": [bool(raised) for raised in m.hand_raised],
            "joints": joints,
            "events": [
                {"id": e.id, "type": e.kind, "side": e.side or "", "strength": _round(e.strength), "time": _round(e.time)}
                for e in self.events
            ],
        }


def _round(value):
    if np.ndim(value):
        return [round(float(v), 4) for v in value]
    return round(float(value), 4)


class PlayerTracker:
    """Filtering and motion analysis for one person."""

    def __init__(self, track_id: int, config: TrackerConfig, event_ids: Iterator[int]):
        self.id = track_id
        self.last_detected = -np.inf
        self._time: float | None = None
        self._confidence_on = config.filter.confidence_on
        self._event_hold = config.event_hold
        self._event_ids = event_ids
        self._recent_events: list[MotionEvent] = []
        self._filter = KeypointFilter(NUM_JOINTS, config.filter)
        self._motion = MotionAnalyzer(NUM_JOINTS, config.motion)

    @property
    def scale(self) -> float:
        return self._motion.scale or FALLBACK_SCALE

    def predict(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Expected joint positions at time `t` and which of them have an estimate to match against."""
        dt = 0.0 if self._time is None else t - self._time
        return self._filter.position + self._filter.velocity * dt, self._filter.state != JointState.LOST

    def distance(self, pose: RawPose, t: float) -> float:
        """Mean distance in torso lengths between a detection and this track's predicted joints."""
        predicted, usable = self.predict(t)
        common = usable & (pose.confidence >= self._confidence_on)
        if not common.any():
            return np.inf
        return float(np.linalg.norm(pose.points[common] - predicted[common], axis=1).mean() / self.scale)

    def update(self, raw: RawPose | None, t: float) -> PlayerPose:
        self._time = t
        scale = self._motion.scale
        if scale is None and raw is not None:
            scale = torso_length(raw.points, raw.confidence >= self._confidence_on)
        if raw is not None:
            self.last_detected = t
        points, confidence = (raw.points, raw.confidence) if raw is not None else (None, None)
        self._filter.update(points, confidence, t, scale or FALLBACK_SCALE)

        f = self._filter
        motion = self._motion.update(f.position, f.velocity, f.state, t)
        fired = tuple(replace(event, id=next(self._event_ids)) for event in motion.events)
        self._recent_events = [e for e in self._recent_events if t - e.time < self._event_hold] + list(fired)
        return PlayerPose(
            self.id, t, raw is not None, f.position.copy(), f.velocity.copy(), f.state.copy(), raw,
            replace(motion, events=fired), tuple(self._recent_events),
        )


class PoseTracker:
    """Tracks every detected person and returns their smoothed poses each frame."""

    def __init__(self, config: TrackerConfig = TrackerConfig()):
        self.config = config
        self._tracks: dict[int, PlayerTracker] = {}
        self._next_id = 0
        self._event_ids = count()

    def update(self, poses: Sequence[RawPose], t: float, ids: Sequence[int | None] | None = None) -> list[PlayerPose]:
        """Advance all tracks to time `t` (s). Without `ids`, detections are matched to tracks
        here. With `ids` (from the identity module) detection i belongs to player ids[i], and a
        None id discards that detection, e.g. a spectator."""
        if ids is None:
            assigned = self._associate(self._deduplicate(poses), t)
        else:
            if len(ids) != len(poses):
                raise ValueError(f"got {len(ids)} ids for {len(poses)} poses")
            assigned = {track_id: pose for track_id, pose in zip(ids, poses) if track_id is not None}
            if len(assigned) != sum(track_id is not None for track_id in ids):
                raise ValueError(f"duplicate ids {list(ids)}")

        for track_id in assigned.keys() - self._tracks.keys():
            self._tracks[track_id] = PlayerTracker(track_id, self.config, self._event_ids)
            self._next_id = max(self._next_id, track_id + 1)

        results = [track.update(assigned.get(track_id), t) for track_id, track in sorted(self._tracks.items())]
        expired = [tid for tid, track in self._tracks.items() if t - track.last_detected > self.config.track_timeout]
        for track_id in expired:
            del self._tracks[track_id]
        return [result for result in results if result.id not in expired]

    def reset(self) -> None:
        self._tracks.clear()

    def _associate(self, poses: list[RawPose], t: float) -> dict[int, RawPose]:
        tracks = list(self._tracks.values())
        assigned: dict[int, RawPose] = {}
        unmatched = set(range(len(poses)))
        if tracks and poses:
            cost = np.array([[track.distance(pose, t) for track in tracks] for pose in poses])
            rows, cols = linear_sum_assignment(np.minimum(cost, 1e6))
            for row, col in zip(rows, cols):
                if cost[row, col] <= self.config.match_distance:
                    assigned[tracks[col].id] = poses[row]
                    unmatched.discard(row)
        for row in sorted(unmatched):
            assigned[self._next_id] = poses[row]
            self._next_id += 1
        return assigned

    def _deduplicate(self, poses: Sequence[RawPose]) -> list[RawPose]:
        """Drop detections that duplicate a more confident detection of the same person."""
        on = self.config.filter.confidence_on
        kept: list[RawPose] = []
        for pose in sorted(poses, key=lambda p: -p.confidence.mean()):
            scale = torso_length(pose.points, pose.confidence >= on) or FALLBACK_SCALE
            if all(_mean_distance(pose, other, on) / scale > self.config.duplicate_distance for other in kept):
                kept.append(pose)
        return kept


def _mean_distance(a: RawPose, b: RawPose, min_confidence: float) -> float:
    common = (a.confidence >= min_confidence) & (b.confidence >= min_confidence)
    if not common.any():
        return np.inf
    return float(np.linalg.norm(a.points[common] - b.points[common], axis=1).mean())
