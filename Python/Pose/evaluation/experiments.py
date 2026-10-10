"""The four evaluations: summary, parameter sweep, dropout and accuracy. Each returns plain data;
printing and plotting live in report.py."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..filtering import FilterConfig
from ..motion import torso_length
from ..skeleton import ELBOWS, JOINT_NAMES, NUM_JOINTS, WRISTS
from ..tracker import TrackerConfig
from .metrics import contiguous_runs, frame_period, match_events, reference_events
from .recording import FrameRecord, Recording, load_recording
from .replay import MIN_TRACK_DURATION, Analysis, Replay, analyses, joint_metrics, pool, replay, weighted_lag

RECOVERED_ERROR = 0.05  # torso lengths from the gap-free output counts as recovered after a gap
JOINT_GROUPS = {"arms": [*ELBOWS, *WRISTS], "all": list(range(NUM_JOINTS))}
GAP_STRATEGIES = {"hold": 0.0, "damped": FilterConfig.prediction_decay, "constant velocity": float("inf")}


def summarize(path: str, config: TrackerConfig, intervals, cutoff: float) -> tuple[dict, list[Analysis]]:
    recording = load_recording(path)
    result = replay(recording, config)
    items = analyses(result, intervals, cutoff)
    times = np.array([frame.time for frame in recording.frames])
    detect_ms = np.array([frame.detect_ms for frame in recording.frames])
    people = np.bincount([min(len(frame.poses), 2) for frame in recording.frames], minlength=3) / len(recording.frames)

    summary = {
        "recording": Path(path).name,
        "image_size": recording.image_size,
        "frames": len(recording.frames),
        "fps": 1.0 / frame_period(times),
        "fps_low": 1.0 / np.percentile(np.diff(times), 95),
        "brightness": float(np.mean([frame.brightness for frame in recording.frames])),
        "detect_ms": _stats(detect_ms),
        "track_ms": _stats(result.track_ms),
        "people_per_frame": {"0": people[0], "1": people[1], "2+": people[2]},
        "tracks": [{"id": a.track.id, "duration_s": a.track.duration, "still_s": a.still.sum() * frame_period(a.track.time)} for a in items],
    }
    if not items:
        return summary, items

    weights = [len(a.track.time) for a in items]
    metrics = [joint_metrics(a) for a in items]
    per_joint = {key: pool([m[key] for m in metrics], weights) for key in metrics[0]}
    summary["torso_px"] = float(pool([np.nanmedian(a.track.scale) for a in items], weights))
    summary["lag_ms"] = weighted_lag(items) * 1000
    summary["joints"] = {name: {key: values[j] for key, values in per_joint.items()} for j, name in enumerate(JOINT_NAMES)}
    summary["mean"] = {key: _finite_mean(values) for key, values in per_joint.items()}

    totals: dict[str, dict] = {}
    for a in items:
        reference = reference_events(a.track, a.reference, config.motion)
        for key in {_event_key(e) for e in a.track.events + reference}:
            expected = [e for e in reference if _event_key(e) == key]
            match = match_events([e for e in a.track.events if _event_key(e) == key], expected)
            total = totals.setdefault(key, {"reference": 0, "delays": [], "extra": 0, "missed": 0})
            total["reference"] += len(expected)
            total["delays"] += match.delays
            total["extra"] += match.extra
            total["missed"] += match.missed
    summary["events"] = {
        key: {"reference": t["reference"], "matched": len(t["delays"]), "extra": t["extra"], "missed": t["missed"],
              "median_delay_ms": float(np.median(t["delays"]) * 1000) if t["delays"] else None}
        for key, t in sorted(totals.items())
    }
    return summary, items


def sweep(path: str, config: TrackerConfig, intervals, cutoff: float, min_cutoffs, betas) -> list[dict]:
    recording = load_recording(path)
    base = replay(recording, config)
    base_items = analyses(base, intervals, cutoff)
    if not base_items:
        raise SystemExit(f"no track in {path} lasted {MIN_TRACK_DURATION} s")
    rows = []
    for min_cutoff in min_cutoffs:
        for beta in betas:
            variant = replace(config, filter=replace(config.filter, min_cutoff=min_cutoff, beta=beta))
            result = replay(recording, variant, base.assignments)
            items = [Analysis(result.tracks[a.track.id], a.reference, a.still) for a in base_items]
            weights = [len(a.track.time) for a in items]
            metrics = [joint_metrics(a) for a in items]
            rows.append({
                "min_cutoff": min_cutoff,
                "beta": beta,
                "still_jitter": _finite_mean(pool([m["still_jitter_filtered"] for m in metrics], weights)),
                "moving_acceleration": _finite_mean(pool([m["moving_acceleration_filtered"] for m in metrics], weights)),
                "moving_error": _finite_mean(pool([m["moving_error_filtered"] for m in metrics], weights)),
                "lag_ms": weighted_lag(items) * 1000,
            })
    return rows


def with_gaps(recording: Recording, gap: np.ndarray, joints: list[int]) -> Recording:
    frames = [
        FrameRecord(frame.index, frame.time, [pose.hide(joints) for pose in frame.poses], frame.detect_ms, frame.brightness)
        if hidden else frame
        for frame, hidden in zip(recording.frames, gap)
    ]
    return Recording(recording.image_size, frames, recording.meta)


def dropout(path: str, config: TrackerConfig, joints: list[int], durations: list[float]) -> list[dict]:
    recording = load_recording(path)
    base = replay(recording, config)
    period = frame_period(np.array([frame.time for frame in recording.frames]))
    rows = []
    for strategy, decay in GAP_STRATEGIES.items():
        variant = replace(config, filter=replace(config.filter, prediction_decay=decay))
        clean = replay(recording, variant, base.assignments)
        for duration in durations:
            length = max(1, round(duration / period))
            spacing = max(3 * length, round(1.5 / period))
            gap = np.zeros(len(recording.frames), dtype=bool)
            for start in range(spacing // 2, len(recording.frames) - length, spacing):
                gap[start : start + length] = True
            gapped = replay(with_gaps(recording, gap, joints), variant, base.assignments)
            rows.append({"strategy": strategy, "gap_ms": length * period * 1000,
                         **_compare_gapped(clean, gapped, gap, joints, period)})
    return rows


def _compare_gapped(clean: Replay, gapped: Replay, gap: np.ndarray, joints: list[int], period: float) -> dict:
    errors, usable, recovery = [], [], []
    extra = gaps = 0
    for track_id, reference in clean.tracks.items():
        if reference.duration < MIN_TRACK_DURATION:
            continue
        track = gapped.tracks[track_id]
        in_gap = gap[reference.frames]
        error = np.linalg.norm(track.position[:, joints] - reference.position[:, joints], axis=2) / reference.scale[:, None]
        checked = in_gap[:, None] & reference.tracked[:, joints]
        errors += error[checked].tolist()
        usable += track.usable[:, joints][checked].tolist()

        for run in contiguous_runs(in_gap):
            if not checked[run].any():
                continue
            gaps += 1
            for k in range(run.stop, len(in_gap)):
                settled = track.usable[k, joints] & (error[k] < RECOVERED_ERROR)
                if (settled | ~reference.tracked[k, joints]).all():
                    recovery.append((k - run.stop + 1) * period)
                    break
        extra += match_events(track.events, reference.events, tolerance=0.3).extra
    errors = np.array(errors)
    return {
        "gaps": gaps,
        "usable": float(np.mean(usable)) if usable else np.nan,
        "median_error": float(np.nanmedian(errors)) if errors.size else np.nan,
        "p90_error": float(np.nanpercentile(errors, 90)) if errors.size else np.nan,
        "recovery_ms": float(np.median(recovery) * 1000) if recovery else np.nan,
        "extra_events": extra,
    }


def accuracy(path: str, annotation_path: str, config: TrackerConfig) -> dict:
    """Compare raw and filtered joints with hand-annotated positions. A joint counts as correct
    (PCK) when it is detected within 20% of the annotated torso length."""
    recording = load_recording(path)
    annotations = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    result = replay(recording, config)
    on = config.filter.confidence_on
    rows = {track_id: {f: row for row, f in enumerate(track.frames)} for track_id, track in result.tracks.items()}
    frame_of = {frame.index: f for f, frame in enumerate(recording.frames)}

    samples = defaultdict(list)  # joint -> (detected, raw error, filtered error, raw error px)
    annotated_people = matched_people = 0
    for key, people in annotations["frames"].items():
        f = frame_of.get(int(key))
        if f is None:
            continue
        poses = recording.frames[f].poses
        for person in people:
            truth = np.full((NUM_JOINTS, 2), np.nan)
            for name, point in person.items():
                if point is not None:
                    truth[JOINT_NAMES.index(name)] = point
            annotated = np.isfinite(truth[:, 0])
            if not annotated.any():
                continue
            annotated_people += 1
            torso = torso_length(truth, annotated)
            distances = [np.linalg.norm(pose.points[annotated] - truth[annotated], axis=1).mean() for pose in poses]
            if not distances or torso is None or min(distances) > torso:
                continue
            matched_people += 1
            best = int(np.argmin(distances))
            pose = poses[best]
            track_id = result.assignments[f][best]
            track = result.tracks.get(track_id)
            for j in np.flatnonzero(annotated):
                raw_px = float(np.linalg.norm(pose.points[j] - truth[j]))
                filtered = np.nan
                if track is not None and f in rows[track_id] and track.usable[rows[track_id][f], j]:
                    filtered = float(np.linalg.norm(track.position[rows[track_id][f], j] - truth[j])) / torso
                samples[int(j)].append((pose.confidence[j] >= on, raw_px / torso, filtered, raw_px))

    joints = {}
    for j, values in sorted(samples.items()):
        detected, raw, filtered, raw_px = (np.array(column, dtype=float) for column in zip(*values))
        hit = detected.astype(bool)
        joints[JOINT_NAMES[j]] = {
            "annotated": len(values),
            "detection_rate": float(hit.mean()),
            "median_error_px": float(np.median(raw_px[hit])) if hit.any() else np.nan,
            "mean_error_raw": float(np.mean(raw[hit])) if hit.any() else np.nan,
            "mean_error_filtered": float(np.nanmean(filtered)) if np.isfinite(filtered).any() else np.nan,
            "pck_raw": float(np.mean(hit & (raw < 0.2))),
            "pck_filtered": float(np.mean(np.nan_to_num(filtered, nan=np.inf) < 0.2)),
        }
    return {"annotated_people": annotated_people, "matched_people": matched_people, "joints": joints}


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else np.nan


def _event_key(event) -> str:
    return f"{event.kind} {event.side}" if event.side else event.kind


def _stats(values: np.ndarray) -> dict:
    return {"mean": float(np.mean(values)), "p50": float(np.median(values)), "p95": float(np.percentile(values, 95))}
