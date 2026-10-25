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

from ..filtering import JointState
from ..motion import MotionAnalyzer, MotionConfig, MotionEvent

MIN_REFERENCE_RUN = 12  # frames; shorter runs of detections are too short to filter forward-backward
MAX_GAP_PERIODS = 3.0  # longer sampling gaps split interpolation and differentiation


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


def run_duration(time: np.ndarray, run: slice) -> float:
    """Seconds covered by the frames of `run`, each frame lasting one frame period: from the first
    frame to one period past the last, so dropped frames inside the run count as elapsed time."""
    return float(time[run.stop - 1] - time[run.start] + frame_period(time))


def _sample_runs(time: np.ndarray, valid: np.ndarray, min_length: int = 2) -> list[slice]:
    limit = MAX_GAP_PERIODS * frame_period(time) * (1 + 1e-9)
    runs = []
    for run in contiguous_runs(valid):
        breaks = np.flatnonzero(np.diff(time[run]) > limit) + run.start + 1
        bounds = [run.start, *breaks, run.stop]
        runs.extend(slice(start, stop) for start, stop in zip(bounds, bounds[1:]) if stop - start >= min_length)
    return runs


def _uniform_time(time: np.ndarray) -> np.ndarray:
    count = max(2, round((time[-1] - time[0]) / frame_period(time)) + 1)
    return np.linspace(time[0], time[-1], count)


def _interpolate(time: np.ndarray, points: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(target, time, coordinate) for coordinate in points.T])


def _resample(time: np.ndarray, points: np.ndarray, valid: np.ndarray, grid: np.ndarray) -> np.ndarray:
    sampled = np.full((len(grid), *points.shape[1:]), np.nan)
    for joint in range(points.shape[1]):
        observed = valid[:, joint] & np.isfinite(points[:, joint]).all(axis=1)
        for run in _sample_runs(time, observed):
            inside = (grid >= time[run.start]) & (grid <= time[run.stop - 1])
            sampled[inside, joint] = _interpolate(time[run], points[run, joint], grid[inside])
    return sampled


def reference_trajectory(track: Track, cutoff: float = 6.0) -> np.ndarray:
    """Zero-phase 2nd-order Butterworth low-pass of each joint's raw trajectory, applied to runs of
    consecutive detections on a uniform time grid, then sampled at the original timestamps."""
    reference = np.full_like(track.raw, np.nan)
    if len(track.time) < MIN_REFERENCE_RUN:
        return reference
    for joint in range(track.raw.shape[1]):
        observed = track.confident[:, joint] & np.isfinite(track.raw[:, joint]).all(axis=1)
        for run in _sample_runs(track.time, observed, MIN_REFERENCE_RUN):
            time = track.time[run]
            grid = _uniform_time(time)
            if len(grid) < MIN_REFERENCE_RUN:
                continue
            rate = 1.0 / frame_period(grid)
            b, a = butter(2, min(cutoff, 0.45 * rate) / (rate / 2))
            sampled = _interpolate(time, track.raw[run, joint], grid)
            filtered = filtfilt(b, a, sampled, axis=0)
            reference[run, joint] = _interpolate(grid, filtered, time)
    return reference


def static_mask(track: Track, reference: np.ndarray, max_speed: float = 0.15, min_duration: float = 1.0) -> np.ndarray:
    """Frames inside periods of at least `min_duration` s in which every observed joint moves slower
    than `max_speed` torso lengths/s, i.e. the player stands still."""
    velocity = np.gradient(reference, track.time, axis=0)
    speed = np.linalg.norm(velocity, axis=2) / track.scale[:, None]
    observed = np.isfinite(speed)
    still = np.where(observed, speed < max_speed, True).all(axis=1) & (observed.sum(axis=1) >= 4)
    mask = np.zeros(len(still), dtype=bool)
    for run in contiguous_runs(still):
        if run_duration(track.time, run) >= min_duration * (1 - 1e-9):
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
    if len(track.time) < 3:
        return np.full(points.shape[1], np.nan)
    dt = np.diff(track.time)
    interval = (dt[:-1] + dt[1:]) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        velocity = np.diff(points, axis=0) / dt[:, None, None]
        acceleration = np.diff(velocity, axis=0) / interval[:, None, None] / track.scale[1:-1, None, None]
    ok = valid[2:] & valid[1:-1] & valid[:-2] & mask[1:-1, None] & np.isfinite(track.scale[1:-1, None])
    regular = (dt > 0) & (dt <= MAX_GAP_PERIODS * frame_period(track.time) * (1 + 1e-9))
    ok &= regular[:-1, None] & regular[1:, None] & np.isfinite(acceleration).all(axis=2)
    squared = np.where(ok, np.sum(acceleration**2, axis=2), 0.0)
    weights = np.where(ok, interval[:, None], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt((squared * weights).sum(axis=0) / weights.sum(axis=0))


def reference_error(track: Track, points: np.ndarray, valid: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per joint RMS distance to the reference in torso lengths (noise plus lag)."""
    ok = valid & np.isfinite(reference[..., 0]) & mask[:, None] & np.isfinite(track.scale[:, None])
    distance = np.linalg.norm(np.nan_to_num(points - reference), axis=2) / track.scale[:, None]
    squared = np.where(ok, distance**2, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(squared.sum(axis=0) / ok.sum(axis=0))


def lag(track: Track, reference: np.ndarray, max_lag: float = 0.3) -> float:
    """Delay in seconds of the filtered output behind the reference: the time shift that best aligns
    them on a uniform time grid, refined with a parabola through the three best shifts.
    Returns nan when there is insufficient motion or no usable alignment."""
    if len(track.time) < 2:
        return np.nan
    dt = frame_period(track.time)
    if not np.isfinite(dt) or dt <= 0:
        return np.nan
    grid = _uniform_time(track.time)
    dt = frame_period(grid)
    position = _resample(track.time, track.position / track.scale[:, None, None], track.tracked, grid)
    target = _resample(track.time, reference / track.scale[:, None, None], np.isfinite(reference).all(axis=2), grid)
    tracked = np.isfinite(position).all(axis=2)
    observed = np.isfinite(target).all(axis=2)
    adjacent = observed[1:] & observed[:-1]
    displacement = np.linalg.norm(np.diff(target, axis=0), axis=2)
    if not np.any(adjacent & (displacement > 1e-6)):
        return np.nan
    max_shift = min(int(max_lag / dt) + 1, len(grid) - 1)
    shifts = np.arange(-max_shift, max_shift + 1)
    errors = []
    for shift in shifts:
        # Compare the output at frame i + shift with the reference at frame i.
        a, b, valid = position[max(shift, 0) :], target[max(-shift, 0) :], tracked[max(shift, 0) :]
        n = min(len(a), len(b))
        ok = valid[:n] & np.isfinite(a[:n]).all(axis=2) & np.isfinite(b[:n]).all(axis=2)
        errors.append(np.sum((a[:n] - b[:n]) ** 2, axis=2)[ok].mean() if ok.any() else np.inf)
    errors = np.array(errors)
    finite = errors[np.isfinite(errors)]
    if len(finite) < 3 or np.allclose(finite, finite[0], rtol=1e-9, atol=1e-12):
        return np.nan
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
