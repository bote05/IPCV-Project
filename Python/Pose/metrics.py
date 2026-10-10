"""Evaluation metrics for replayed recordings.

Live sessions have no ground truth, so smoothness and lag are measured against a reference: a
zero-phase (forward-backward) Butterworth low-pass of the raw detections. It follows the motion
without delay but needs future frames, so it is the best a filter could do offline; the distance of
the live filter from it shows the jitter and lag the live filter leaves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, filtfilt

from .filtering import JointState
from .motion import MotionAnalyzer, MotionConfig, MotionEvent

MIN_REFERENCE_RUN = 12  # frames; shorter runs of detections are too short to filter forward-backward


@dataclass
class Track:
    """One tracked person over the frames in which the track existed."""

    id: int
    frames: np.ndarray  # (F,) indices into the recording's frames
    time: np.ndarray  # (F,) s
    detected: np.ndarray  # (F,) bool
    raw: np.ndarray  # (F, N, 2) px, nan in frames where the person was not detected
    confidence: np.ndarray  # (F, N), 0 in frames where the person was not detected
    position: np.ndarray  # (F, N, 2) filtered, px
    state: np.ndarray  # (F, N) JointState
    scale: np.ndarray  # (F,) torso length in px, nan before the first estimate
    min_confidence: float  # detections below this count as missing
    events: list[MotionEvent] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return float(self.time[-1] - self.time[0]) if len(self.time) > 1 else 0.0

    @property
    def confident(self) -> np.ndarray:
        return self.confidence >= self.min_confidence

    @property
    def tracked(self) -> np.ndarray:
        return self.state == JointState.TRACKED

    @property
    def usable(self) -> np.ndarray:
        return (self.state == JointState.TRACKED) | (self.state == JointState.PREDICTED)


def contiguous_runs(mask: np.ndarray, min_length: int = 1) -> list[slice]:
    """Slices covering runs of consecutive True values at least `min_length` long."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return [slice(start, stop) for start, stop in zip(edges[::2], edges[1::2]) if stop - start >= min_length]


def frame_period(time: np.ndarray) -> float:
    return float(np.median(np.diff(time)))


def reference_trajectory(track: Track, cutoff: float = 6.0) -> np.ndarray:
    """Zero-phase 2nd-order Butterworth low-pass of each joint's raw trajectory, applied to runs of
    consecutive detections; nan elsewhere."""
    rate = 1.0 / frame_period(track.time)
    b, a = butter(2, min(cutoff, 0.45 * rate) / (rate / 2))
    reference = np.full_like(track.raw, np.nan)
    for joint in range(track.raw.shape[1]):
        for run in contiguous_runs(track.confident[:, joint], MIN_REFERENCE_RUN):
            reference[run, joint] = filtfilt(b, a, track.raw[run, joint], axis=0)
    return reference


def static_mask(track: Track, reference: np.ndarray, max_speed: float = 0.15, min_duration: float = 1.0) -> np.ndarray:
    """Frames inside periods of at least `min_duration` s in which every observed joint moves slower
    than `max_speed` torso lengths/s, i.e. the player stands still."""
    velocity = np.gradient(reference, track.time, axis=0)
    speed = np.linalg.norm(velocity, axis=2) / track.scale[:, None]
    observed = np.isfinite(speed)
    still = np.where(observed, speed < max_speed, True).all(axis=1) & (observed.sum(axis=1) >= 4)
    mask = np.zeros(len(still), dtype=bool)
    for run in contiguous_runs(still, round(min_duration / frame_period(track.time))):
        mask[run] = True
    return mask


def interval_mask(time: np.ndarray, intervals: list[tuple[float, float]]) -> np.ndarray:
    mask = np.zeros(len(time), dtype=bool)
    for start, stop in intervals:
        mask |= (time >= start) & (time <= stop)
    return mask


def detection_rate(track: Track) -> np.ndarray:
    """Per joint: fraction of frames with a detection of the person in which the joint was confident."""
    if not track.detected.any():
        return np.full(track.raw.shape[1], np.nan)
    return track.confident[track.detected].mean(axis=0)


def availability(track: Track) -> np.ndarray:
    """Per joint: fraction of the track's frames in which the filter output was usable."""
    return track.usable.mean(axis=0)


def static_jitter(track: Track, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per joint RMS deviation from the mean position within each still period, in torso lengths,
    for the raw detections and the filtered output."""
    num_joints = track.raw.shape[1]
    squares, counts = np.zeros((2, num_joints)), np.zeros((2, num_joints))
    for run in contiguous_runs(mask):
        scale = np.nanmean(track.scale[run])
        series = ((track.raw[run], track.confident[run]), (track.position[run], track.tracked[run]))
        for k, (points, valid) in enumerate(series):
            for joint in range(num_joints):
                samples = points[valid[:, joint], joint]
                if len(samples) >= 2:
                    squares[k, joint] += np.sum((samples - samples.mean(axis=0)) ** 2) / scale**2
                    counts[k, joint] += len(samples)
    with np.errstate(invalid="ignore", divide="ignore"):
        rms = np.sqrt(squares / counts)
    return rms[0], rms[1]


def acceleration_rms(track: Track, points: np.ndarray, valid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per joint RMS of the second time derivative in torso lengths/s², a jitter measure that also
    applies while moving: smooth motion has low acceleration, frame-to-frame noise has very high."""
    dt = frame_period(track.time)
    acceleration = (points[2:] - 2 * points[1:-1] + points[:-2]) / dt**2 / track.scale[1:-1, None, None]
    ok = valid[2:] & valid[1:-1] & valid[:-2] & mask[1:-1, None] & np.isfinite(track.scale[1:-1, None])
    squared = np.where(ok, np.sum(np.nan_to_num(acceleration) ** 2, axis=2), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(squared.sum(axis=0) / ok.sum(axis=0))


def reference_error(track: Track, points: np.ndarray, valid: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per joint RMS distance to the reference in torso lengths (noise plus lag)."""
    ok = valid & np.isfinite(reference[..., 0]) & mask[:, None] & np.isfinite(track.scale[:, None])
    distance = np.linalg.norm(np.nan_to_num(points - reference), axis=2) / track.scale[:, None]
    squared = np.where(ok, distance**2, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(squared.sum(axis=0) / ok.sum(axis=0))


def lag(track: Track, reference: np.ndarray, max_lag: float = 0.3) -> float:
    """Delay in seconds of the filtered output behind the reference: the time shift that best aligns
    them, refined to sub-frame precision with a parabola through the three best shifts."""
    dt = frame_period(track.time)
    position = track.position / track.scale[:, None, None]
    target = reference / track.scale[:, None, None]
    tracked = track.tracked
    max_shift = int(max_lag / dt) + 1
    shifts = np.arange(-max_shift, max_shift + 1)
    errors = []
    for shift in shifts:
        # Compare the output at frame i + shift with the reference at frame i.
        a, b, valid = position[max(shift, 0) :], target[max(-shift, 0) :], tracked[max(shift, 0) :]
        n = min(len(a), len(b))
        ok = valid[:n] & np.isfinite(a[:n, :, 0]) & np.isfinite(b[:n, :, 0])
        errors.append(np.sum((a[:n] - b[:n]) ** 2, axis=2)[ok].mean() if ok.any() else np.inf)
    errors = np.array(errors)
    best = int(np.argmin(errors))
    offset = 0.0
    if 0 < best < len(errors) - 1 and np.isfinite(errors[best - 1 : best + 2]).all():
        left, middle, right = errors[best - 1 : best + 2]
        curvature = left - 2 * middle + right
        if curvature > 0:
            offset = 0.5 * (left - right) / curvature
    return float((shifts[best] + offset) * dt)


def reference_events(track: Track, reference: np.ndarray, config: MotionConfig) -> list[MotionEvent]:
    """Motion events detected on the zero-phase reference: when the movement really happened."""
    analyzer = MotionAnalyzer(reference.shape[1], config)
    velocity = np.gradient(reference, track.time, axis=0)
    valid = np.isfinite(reference[..., 0]) & np.isfinite(velocity[..., 0])
    events: list[MotionEvent] = []
    for f, t in enumerate(track.time):
        state = np.where(valid[f], JointState.TRACKED, JointState.LOST)
        events += analyzer.update(np.nan_to_num(reference[f]), np.nan_to_num(velocity[f]), state, t).events
    return events


@dataclass
class EventMatch:
    delays: list[float]  # s, online event time minus reference event time
    extra: int  # online events with no reference counterpart (false triggers)
    missed: int  # reference events the online pipeline did not produce


def match_events(online: list[MotionEvent], reference: list[MotionEvent], tolerance: float = 0.5) -> EventMatch:
    unmatched = list(reference)
    delays = []
    for event in online:
        candidates = [r for r in unmatched if r.kind == event.kind and r.side == event.side and abs(event.time - r.time) <= tolerance]
        if candidates:
            nearest = min(candidates, key=lambda r: abs(event.time - r.time))
            unmatched.remove(nearest)
            delays.append(event.time - nearest.time)
    return EventMatch(delays, len(online) - len(delays), len(unmatched))
