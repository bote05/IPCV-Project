"""Per-player pose tracking: keeps each person's filter state attached to the same person across
frames and turns their detections into smoothed joints and motion signals.

MediaPipe returns people in no guaranteed order, so detections are matched to existing tracks by
their distance to each track's predicted joints (Hungarian assignment). This short-term matching is
only what temporal filtering needs; mapping tracks to persistent players across crossings, long
occlusions and re-entry is the identity module's job, which can pass its own ids to `update`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detector import RawPose
from .filtering import FilterConfig, JointState, KeypointFilter
from .motion import MotionAnalyzer, MotionConfig, MotionSignals, torso_length
from .skeleton import JOINT_NAMES, NUM_JOINTS

FALLBACK_SCALE = 150.0  # px, torso length of a player ~2 m from a 720p webcam; used until measured


@dataclass(frozen=True)
class TrackerConfig:
    match_distance: float = 1.0  # torso lengths; mean joint distance beyond which a detection is a new person
    duplicate_distance: float = 0.2  # torso lengths; detections closer than this are the same person
    track_timeout: float = 1.0  # s without a detection before a track is dropped
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
    motion: MotionSignals

    def to_dict(self, image_size: tuple[int, int]) -> dict:
        """JSON-ready summary for the game. Image coordinates are normalised to [0, 1] (origin top
        left); body-frame values are in torso lengths with y up. Lists of objects instead of maps
        keep it parseable by Unity's JsonUtility."""
        width, height = image_size
        m = self.motion
        joints = [
            {
                "name": name,
                "state": int(self.state[j]),
                "x": _round(self.position[j, 0] / width),
                "y": _round(self.position[j, 1] / height),
                "body": _round(m.body[j]),
                "velocity": _round(m.body_velocity[j]),
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
            "center": [_round(m.center[0] / width), _round(m.center[1] / height)],
            "center_velocity": _round(m.center_velocity),
            "lean": _round(m.lean),
            "arm_extension": _round(m.arm_extension),
            "hand_raised": [bool(raised) for raised in m.hand_raised],
            "joints": joints,
            "events": [{"type": e.kind, "side": e.side or "", "strength": _round(e.strength)} for e in m.events],
        }


def _round(value):
    if np.ndim(value):
        return [round(float(v), 4) for v in value]
    return round(float(value), 4)


class PlayerTracker:
    """Filtering and motion analysis for one person."""

    def __init__(self, track_id: int, config: TrackerConfig):
        self.id = track_id
        self.last_detected = -np.inf
        self._time: float | None = None
        self._confidence_on = config.filter.confidence_on
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
        return PlayerPose(self.id, t, raw is not None, f.position.copy(), f.velocity.copy(), f.state.copy(), raw, motion)


class PoseTracker:
    """Tracks every detected person and returns their smoothed poses each frame."""

    def __init__(self, config: TrackerConfig = TrackerConfig()):
        self.config = config
        self._tracks: dict[int, PlayerTracker] = {}
        self._next_id = 0

    def update(self, poses: Sequence[RawPose], t: float, ids: Sequence[int] | None = None) -> list[PlayerPose]:
        """Advance all tracks to time `t` (s). Without `ids`, detections are matched to tracks
        here; with `ids` (e.g. from the identity module) detection i belongs to track ids[i]."""
        if ids is None:
            assigned = self._associate(self._deduplicate(poses), t)
        else:
            assigned = dict(zip(ids, poses))

        for track_id in assigned.keys() - self._tracks.keys():
            self._tracks[track_id] = PlayerTracker(track_id, self.config)
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
