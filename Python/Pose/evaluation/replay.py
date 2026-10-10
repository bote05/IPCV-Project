"""Replaying recorded detections through the tracker, and per-track analysis of the result."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from ..skeleton import NUM_JOINTS
from ..tracker import PoseTracker, TrackerConfig
from .metrics import (
    Track,
    acceleration_rms,
    availability,
    detection_rate,
    interval_mask,
    lag,
    reference_error,
    reference_trajectory,
    static_jitter,
    static_mask,
)
from .recording import Recording

MIN_TRACK_DURATION = 2.0  # s; shorter tracks (people passing by, false detections) are not evaluated


@dataclass
class Replay:
    tracks: dict[int, Track]
    assignments: list[list[int]]  # per frame, the track id each recorded pose was given (-1: discarded)
    track_ms: np.ndarray


@dataclass
class Analysis:
    track: Track
    reference: np.ndarray
    still: np.ndarray


def replay(recording: Recording, config: TrackerConfig, assignments: list[list[int]] | None = None) -> Replay:
    """Run the tracker over a recording. With `assignments` (from an earlier replay) every pose goes
    to the same track as before, so runs with different settings stay comparable frame by frame."""
    tracker = PoseTracker(config)
    on = config.filter.confidence_on
    rows: dict[int, list] = defaultdict(list)
    events: dict[int, list] = defaultdict(list)
    used, timings = [], []
    for f, frame in enumerate(recording.frames):
        poses, ids = frame.poses, None
        if assignments is not None:
            kept = [i for i, track_id in enumerate(assignments[f]) if track_id >= 0]
            poses, ids = [frame.poses[i] for i in kept], [assignments[f][i] for i in kept]
        tic = time.perf_counter()
        players = tracker.update(poses, frame.time, ids)
        timings.append((time.perf_counter() - tic) * 1000)

        owner = {id(player.raw): player.id for player in players if player.raw is not None}
        used.append([owner.get(id(pose), -1) for pose in frame.poses])
        for player in players:
            raw, confidence = np.full((NUM_JOINTS, 2), np.nan), np.zeros(NUM_JOINTS)
            if player.raw is not None:
                raw, confidence = player.raw.points, player.raw.confidence
            scale = player.motion.scale or np.nan
            rows[player.id].append((f, frame.time, player.detected, raw, confidence, player.position, player.state, scale))
            events[player.id] += player.motion.events

    tracks = {
        track_id: Track(track_id, *(np.array(column) for column in zip(*track_rows)), min_confidence=on, events=events[track_id])
        for track_id, track_rows in rows.items()
    }
    return Replay(tracks, used, np.array(timings))


def analyses(result: Replay, intervals: list[tuple[float, float]], cutoff: float) -> list[Analysis]:
    out = []
    for track in result.tracks.values():
        if track.duration < MIN_TRACK_DURATION:
            continue
        reference = reference_trajectory(track, cutoff)
        still = interval_mask(track.time, intervals) if intervals else static_mask(track, reference)
        out.append(Analysis(track, reference, still))
    return out


def joint_metrics(a: Analysis) -> dict[str, np.ndarray]:
    track, moving = a.track, ~a.still
    raw_valid = track.confident
    jitter_raw, jitter_filtered = static_jitter(track, a.still)
    return {
        "detection_rate": detection_rate(track),
        "availability": availability(track),
        "still_jitter_raw": jitter_raw,
        "still_jitter_filtered": jitter_filtered,
        "moving_acceleration_raw": acceleration_rms(track, track.raw, raw_valid, moving),
        "moving_acceleration_filtered": acceleration_rms(track, track.position, track.tracked, moving),
        "moving_error_raw": reference_error(track, track.raw, raw_valid, a.reference, moving),
        "moving_error_filtered": reference_error(track, track.position, track.tracked, a.reference, moving),
    }


def pool(values: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Weighted mean over tracks, skipping tracks without data for an entry."""
    stacked = np.array(values, dtype=float)
    w = np.where(np.isfinite(stacked), np.reshape(weights, (-1,) + (1,) * (stacked.ndim - 1)), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nansum(stacked * w, axis=0) / w.sum(axis=0)


def weighted_lag(items: list[Analysis]) -> float:
    return float(pool([lag(a.track, a.reference) for a in items], [len(a.track.time) for a in items]))
