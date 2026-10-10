"""Offline evaluation of the pose module on recorded sessions (record with demo.py --record).

    python -m Pose.evaluate summary recordings/*.jsonl [--static 0-5] [--plots out] [--json out.json]
    python -m Pose.evaluate sweep recordings/s1.jsonl
    python -m Pose.evaluate dropout recordings/s1.jsonl
    python -m Pose.evaluate accuracy recordings/s1.jsonl recordings/s1_gt.json

summary   detection rate, jitter (still and moving), lag, event delay, latency; one table per
          recording plus a comparison across recordings (e.g. lighting conditions)
sweep     jitter/lag trade-off over a grid of One Euro parameters
dropout   hides joints for gaps of increasing length and measures prediction error, recovery time
          and false events, for three gap-handling strategies
accuracy  keypoint error against hand-annotated frames (see annotate.py)

Each command replays the recorded raw detections through the tracker, so every setting is
evaluated on identical input.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .filtering import FilterConfig, JointState
from .metrics import (
    Track,
    acceleration_rms,
    availability,
    contiguous_runs,
    detection_rate,
    frame_period,
    interval_mask,
    lag,
    match_events,
    reference_error,
    reference_events,
    reference_trajectory,
    static_jitter,
    static_mask,
)
from .motion import torso_length
from .recording import FrameRecord, Recording, load_recording
from .skeleton import ELBOWS, JOINT_NAMES, NUM_JOINTS, WRISTS, Joint
from .tracker import PoseTracker, TrackerConfig

MIN_TRACK_DURATION = 2.0  # s; shorter tracks (people passing by, false detections) are not evaluated
RECOVERED_ERROR = 0.05  # torso lengths from the gap-free output counts as recovered after a gap
JOINT_GROUPS = {"arms": [*ELBOWS, *WRISTS], "all": list(range(NUM_JOINTS))}
GAP_STRATEGIES = {"hold": 0.0, "damped": FilterConfig.prediction_decay, "constant velocity": float("inf")}


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


# --------------------------------------------------------------------------------------- summary


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


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else np.nan


def _event_key(event) -> str:
    return f"{event.kind} {event.side}" if event.side else event.kind


def _stats(values: np.ndarray) -> dict:
    return {"mean": float(np.mean(values)), "p50": float(np.median(values)), "p95": float(np.percentile(values, 95))}


def print_summary(s: dict) -> None:
    width, height = s["image_size"]
    print(f"\n== {s['recording']}  {width}x{height}, {s['frames']} frames, {s['fps']:.1f} fps "
          f"(5th percentile {s['fps_low']:.1f}), brightness {s['brightness']:.0f}/255")
    print(f"detection {s['detect_ms']['mean']:.1f} ms mean / {s['detect_ms']['p95']:.1f} ms p95, "
          f"tracking {s['track_ms']['mean']:.2f} ms mean / {s['track_ms']['p95']:.2f} ms p95")
    p = s["people_per_frame"]
    print(f"people per frame: 0: {p['0']:.0%}  1: {p['1']:.0%}  2+: {p['2+']:.0%}")
    print("tracks: " + ", ".join(f"#{t['id']} {t['duration_s']:.1f} s ({t['still_s']:.1f} s still)" for t in s["tracks"]))
    if "joints" not in s:
        print(f"no track lasted {MIN_TRACK_DURATION} s, nothing to evaluate")
        return
    print(f"median torso length {s['torso_px']:.0f} px, filter lag {s['lag_ms']:.0f} ms behind the zero-phase reference\n")

    headers = ["joint", "detected", "usable", "still jitter %torso", "moving accel torso/s2", "moving error %torso"]
    print(f"{headers[0]:<15}{headers[1]:>9}{headers[2]:>8}   {headers[3]:<21}{headers[4]:<24}{headers[5]}")
    print(f"{'':<15}{'':>9}{'':>8}   {'raw':>8}{'filtered':>10}   {'raw':>8}{'filtered':>10}      {'raw':>8}{'filtered':>10}")
    for name, m in [*s["joints"].items(), ("mean", s["mean"])]:
        print(f"{name:<15}{m['detection_rate']:>9.0%}{m['availability']:>8.0%}   "
              f"{100 * m['still_jitter_raw']:>8.2f}{100 * m['still_jitter_filtered']:>10.2f}   "
              f"{m['moving_acceleration_raw']:>8.1f}{m['moving_acceleration_filtered']:>10.1f}      "
              f"{100 * m['moving_error_raw']:>8.2f}{100 * m['moving_error_filtered']:>10.2f}")

    if s["events"]:
        print("\nevents vs zero-phase reference:")
        for key, e in s["events"].items():
            delay = f"{e['median_delay_ms']:.0f} ms median delay" if e["median_delay_ms"] is not None else "no matches"
            print(f"  {key:<12} {e['matched']}/{e['reference']} matched, {delay}, {e['extra']} extra, {e['missed']} missed")


def print_comparison(summaries: list[dict]) -> None:
    print("\n== comparison")
    print(f"{'recording':<28}{'bright':>7}{'fps':>6}{'det ms':>8}{'2 people':>9}{'detected':>9}"
          f"{'jitter raw':>11}{'filtered':>9}{'lag ms':>7}")
    for s in summaries:
        m = s.get("mean", {})
        print(f"{s['recording'][:27]:<28}{s['brightness']:>7.0f}{s['fps']:>6.1f}{s['detect_ms']['mean']:>8.1f}"
              f"{s['people_per_frame']['2+']:>9.0%}{m.get('detection_rate', np.nan):>9.0%}"
              f"{100 * m.get('still_jitter_raw', np.nan):>11.2f}{100 * m.get('still_jitter_filtered', np.nan):>9.2f}"
              f"{s.get('lag_ms', np.nan):>7.0f}")


def plot_trajectory(a: Analysis, joint: Joint, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    track = a.track
    t = track.time - track.time[0]
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for axis, coordinate in zip(axes, "xy"):
        k = "xy".index(coordinate)
        predicted = track.state[:, joint] == JointState.PREDICTED
        for run in contiguous_runs(predicted):
            axis.axvspan(t[run.start], t[run.stop - 1], color="orange", alpha=0.2, lw=0)
        confident = track.confident[:, joint]
        axis.plot(t[confident], track.raw[confident, joint, k], ".", color="dimgrey", ms=4, label="raw detection")
        axis.plot(t[~confident], track.raw[~confident, joint, k], "o", mfc="none", color="silver", ms=3,
                  label="low-confidence detection")
        axis.plot(t, a.reference[:, joint, k], "--", color="black", lw=1, label="zero-phase reference")
        shown = track.usable[:, joint]
        axis.plot(t, np.where(shown, track.position[:, joint, k], np.nan), color="tab:blue", lw=1.5, label="filtered")
        axis.set_ylabel(f"{coordinate} (px)")
        if coordinate == "y":
            axis.invert_yaxis()
    axes[0].set_title(f"track {track.id}, {joint.name.lower()} (orange: predicted)")
    axes[0].legend(loc="upper right")
    axes[1].set_xlabel("time (s)")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


# ----------------------------------------------------------------------------------------- sweep


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


def print_sweep(rows: list[dict], config: TrackerConfig) -> None:
    print(f"\n{'min_cutoff':>10}{'beta':>6}{'still jitter %torso':>21}{'moving accel':>14}{'moving error %torso':>21}{'lag ms':>8}")
    for r in rows:
        default = " <- default" if (r["min_cutoff"], r["beta"]) == (config.filter.min_cutoff, config.filter.beta) else ""
        print(f"{r['min_cutoff']:>10.2f}{r['beta']:>6.2f}{100 * r['still_jitter']:>21.2f}{r['moving_acceleration']:>14.1f}"
              f"{100 * r['moving_error']:>21.2f}{r['lag_ms']:>8.0f}{default}")


def plot_sweep(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 5))
    for beta in sorted({r["beta"] for r in rows}):
        subset = [r for r in rows if r["beta"] == beta]
        axis.plot([r["lag_ms"] for r in subset], [100 * r["still_jitter"] for r in subset], "o-", label=f"beta {beta:g}")
        for r in subset:
            axis.annotate(f"{r['min_cutoff']:g}", (r["lag_ms"], 100 * r["still_jitter"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axis.set_xlabel("lag behind reference (ms)")
    axis.set_ylabel("jitter while still (% torso)")
    axis.set_title("One Euro trade-off (labels: min_cutoff Hz)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


# --------------------------------------------------------------------------------------- dropout


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


def print_dropout(rows: list[dict], group: str) -> None:
    print(f"\nhidden joints: {group}")
    print(f"{'strategy':<19}{'gap ms':>7}{'gaps':>6}{'usable':>8}{'median err %torso':>19}{'p90 err':>9}{'recovery ms':>13}{'false events':>14}")
    for r in rows:
        print(f"{r['strategy']:<19}{r['gap_ms']:>7.0f}{r['gaps']:>6}{r['usable']:>8.0%}{100 * r['median_error']:>19.1f}"
              f"{100 * r['p90_error']:>9.1f}{r['recovery_ms']:>13.0f}{r['extra_events']:>14}")


def plot_dropout(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (error_axis, usable_axis) = plt.subplots(1, 2, figsize=(11, 4))
    for strategy in GAP_STRATEGIES:
        subset = [r for r in rows if r["strategy"] == strategy]
        gap_ms = [r["gap_ms"] for r in subset]
        line, = error_axis.plot(gap_ms, [100 * r["median_error"] for r in subset], "o-", label=f"{strategy} median")
        error_axis.plot(gap_ms, [100 * r["p90_error"] for r in subset], "s--", color=line.get_color(), label=f"{strategy} p90")
        usable_axis.plot(gap_ms, [100 * r["usable"] for r in subset], "o-", label=strategy)
    error_axis.set(xlabel="gap length (ms)", ylabel="error during gap (% torso)", title="prediction error")
    usable_axis.set(xlabel="gap length (ms)", ylabel="joint usable during gap (%)", title="availability")
    error_axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


# -------------------------------------------------------------------------------------- accuracy


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


def print_accuracy(result: dict) -> None:
    print(f"\n{result['matched_people']}/{result['annotated_people']} annotated people matched to a detection")
    print(f"{'joint':<15}{'n':>5}{'detected':>10}{'median px':>11}{'mean err %torso':>17}{'filtered':>10}{'PCK@0.2':>9}{'filtered':>10}")
    for name, m in result["joints"].items():
        print(f"{name:<15}{m['annotated']:>5}{m['detection_rate']:>10.0%}{m['median_error_px']:>11.1f}"
              f"{100 * m['mean_error_raw']:>17.1f}{100 * m['mean_error_filtered']:>10.1f}{m['pck_raw']:>9.0%}{m['pck_filtered']:>10.0%}")


# ------------------------------------------------------------------------------------------- cli


def _interval(text: str) -> tuple[float, float]:
    start, stop = text.split("-")
    return float(start), float(stop)


def _write_json(path: str | None, data) -> None:
    if path:
        Path(path).write_text(json.dumps(data, indent=2, default=_to_builtin), encoding="utf-8")


def _to_builtin(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--min-cutoff", type=float, default=FilterConfig.min_cutoff)
    common.add_argument("--beta", type=float, default=FilterConfig.beta)
    common.add_argument("--static", type=_interval, action="append", default=[], metavar="START-END",
                        help="seconds in which the players stood still (default: detected automatically)")
    common.add_argument("--reference-cutoff", type=float, default=6.0, help="Hz, zero-phase reference low-pass")
    common.add_argument("--json", help="also write the results to this file")
    common.add_argument("--plots", help="directory for figures")

    summary_parser = commands.add_parser("summary", parents=[common])
    summary_parser.add_argument("recordings", nargs="+")
    summary_parser.add_argument("--joint", default="right_wrist", choices=JOINT_NAMES, help="joint for the trajectory plot")
    sweep_parser = commands.add_parser("sweep", parents=[common])
    sweep_parser.add_argument("recording")
    sweep_parser.add_argument("--min-cutoffs", type=float, nargs="+", default=[0.3, 0.5, 1.0, 1.5, 2.5, 4.0])
    sweep_parser.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 4.0])
    dropout_parser = commands.add_parser("dropout", parents=[common])
    dropout_parser.add_argument("recording")
    dropout_parser.add_argument("--joints", choices=JOINT_GROUPS, default="arms")
    dropout_parser.add_argument("--durations", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.8])
    accuracy_parser = commands.add_parser("accuracy", parents=[common])
    accuracy_parser.add_argument("recording")
    accuracy_parser.add_argument("annotations")
    args = parser.parse_args()

    config = TrackerConfig(filter=replace(FilterConfig(), min_cutoff=args.min_cutoff, beta=args.beta))
    plots = Path(args.plots) if args.plots else None
    if plots:
        plots.mkdir(parents=True, exist_ok=True)

    if args.command == "summary":
        results = [summarize(path, config, args.static, args.reference_cutoff) for path in args.recordings]
        summaries = [summary for summary, _ in results]
        for summary in summaries:
            print_summary(summary)
        if len(summaries) > 1:
            print_comparison(summaries)
        if plots:
            joint = Joint[args.joint.upper()]
            for path, (_, items) in zip(args.recordings, results):
                if items:
                    best = max(items, key=lambda a: a.track.confident[:, joint].sum())
                    plot_trajectory(best, joint, plots / f"{Path(path).stem}_trajectory.png")
        _write_json(args.json, summaries)
    elif args.command == "sweep":
        rows = sweep(args.recording, config, args.static, args.reference_cutoff, args.min_cutoffs, args.betas)
        print_sweep(rows, config)
        if plots:
            plot_sweep(rows, plots / f"{Path(args.recording).stem}_sweep.png")
        _write_json(args.json, rows)
    elif args.command == "dropout":
        rows = dropout(args.recording, config, JOINT_GROUPS[args.joints], args.durations)
        print_dropout(rows, args.joints)
        if plots:
            plot_dropout(rows, plots / f"{Path(args.recording).stem}_dropout_{args.joints}.png")
        _write_json(args.json, rows)
    else:
        result = accuracy(args.recording, args.annotations, config)
        print_accuracy(result)
        _write_json(args.json, result)


if __name__ == "__main__":
    main()
