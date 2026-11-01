
"""Evaluate two-player pose identity tracking from timestamped CSV recordings.

Raw poses CSV:
    timestamp,pose_index,joints,confidence

Identified poses CSV:
    timestamp,pose_index,predicted_id,actual_result

Labels:
    A/B = target players
    S = spectator/non-target
    U or blank = uncertain/unreviewed

Pose indices are unique only within a timestamp.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


PLAYER_IDS = {"A": 1, "B": 2}
PLAYERS = {"A", "B"}
NON_PLAYERS = {"S", "SPECTATOR", "OTHER", "NON_PLAYER"}
UNKNOWN = {"", "U", "UNKNOWN", "?"}


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def value(row, *keys):
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def number(v):
    if v is None or v == "":
        return None

    try:
        n = float(v)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def integer(v):
    n = number(v)
    return int(n) if n is not None and n.is_integer() else None


def tracker_id(v):
    if v is None or str(v).strip() == "":
        return None

    try:
        n = int(float(v))
    except (ValueError, TypeError):
        return None

    return n if n in (1, 2) else None


def label(v):
    return "" if v is None else str(v).strip().upper()


def mean(values):
    return float(statistics.mean(values)) if values else None


def median(values):
    return float(statistics.median(values)) if values else None


def percent(n, d):
    return 100.0 * n / d if d else None


def timestamp_key(row):
    """Create a key for matching observations across the two CSV files."""
    ts = number(value(row, "timestamp"))
    pi = integer(value(row, "pose_index"))

    if ts is None or pi is None or pi < 0:
        return None

    return round(ts, 6), pi


def reentry_metrics(identity_by_key, events_csv):
    if not events_csv:
        return {
            "event_count": 0,
            "recovered_correctly": 0,
            "recovery_rate_percent": None,
            "mean_recovery_time_seconds": None,
            "events": [],
            "note": "Supply a re-entry events CSV to evaluate recovery."
        }

    results = []
    recovery_times = []

    for r in read_csv(events_csv):
        person = label(value(r, "person", "true_person"))
        start = number(value(r, "reentry_timestamp", "timestamp"))
        deadline = number(value(r, "deadline_seconds")) or 1.0
        event_id = value(r, "event_id", "id")

        # Format: 3.333:0;3.367:1;3.400:0
        pose_map = {}
        raw_map = value(r, "person_pose_indices", "pose_indices") or ""

        for pair in raw_map.split(";"):
            if ":" not in pair:
                continue

            ts_text, pi_text = pair.split(":", 1)
            ts = number(ts_text.strip())
            pi = integer(pi_text.strip())

            if ts is not None and pi is not None:
                pose_map[round(ts, 6)] = pi

        if person not in PLAYERS or start is None:
            continue

        found = None

        for ts in sorted(pose_map):
            if ts < start or ts > start + deadline:
                continue

            pi = pose_map[ts]
            row = identity_by_key.get((ts, pi))

            if row and tracker_id(row["predicted_id"]) == PLAYER_IDS[person]:
                found = ts
                break

        recovered = found is not None
        elapsed = found - start if recovered else None

        if elapsed is not None:
            recovery_times.append(elapsed)

        results.append({
            "event_id": event_id,
            "person": person,
            "recovered": recovered,
            "reentry_timestamp": start,
            "recovery_timestamp": found,
            "recovery_time_seconds": elapsed
        })

    count = len(results)
    recovered = sum(r["recovered"] for r in results)

    return {
        "event_count": count,
        "recovered_correctly": recovered,
        "recovery_rate_percent": percent(recovered, count),
        "mean_recovery_time_seconds": mean(recovery_times),
        "events": results,
        "note": (
            "Recovery is the first correctly assigned labelled pose "
            "within deadline_seconds after re-entry."
        )
    }


def evaluate_trial(
    raw_poses_csv,
    identified_poses_csv,
    *,
    trial_name=None,
    condition=None,
    crossing_trial=False,
    reentry_events_csv=None,
    output_json=None
):
    raw_rows = read_csv(raw_poses_csv)
    identity_rows = read_csv(identified_poses_csv)

    # Index raw pose observations.
    raw_by_key = {}

    for row in raw_rows:
        key = timestamp_key(row)

        if key is not None:
            raw_by_key[key] = row

    # Index predicted IDs and ground-truth annotations.
    identity_by_key = {}

    for row in identity_rows:
        key = timestamp_key(row)

        if key is not None:
            identity_by_key[key] = {
                "predicted_id": value(row, "predicted_id"),
                "actual_result": label(value(row, "actual_result"))
            }

    counts = {
        p: {"labelled": 0, "correct": 0, "assigned": 0}
        for p in PLAYERS
    }

    player_poses = 0
    correct_poses = 0
    assigned_poses = 0

    nonplayers = 0
    false_assignments = 0
    unknown = 0
    missing_raw_rows = 0
    missing_identity_rows = 0

    by_timestamp = defaultdict(dict)
    observations = defaultdict(list)

    # Evaluate every ground-truth annotation.
    for key, identity in identity_by_key.items():
        ts, pi = key
        lab = identity["actual_result"]
        pred = tracker_id(identity["predicted_id"])

        if lab in UNKNOWN:
            unknown += 1
            continue

        if key not in raw_by_key:
            missing_raw_rows += 1

        if lab in PLAYERS:
            player_poses += 1
            counts[lab]["labelled"] += 1

            expected = PLAYER_IDS[lab]

            if pred is not None:
                assigned_poses += 1
                counts[lab]["assigned"] += 1

            if pred == expected:
                correct_poses += 1
                counts[lab]["correct"] += 1

            by_timestamp[ts][lab] = pred
            observations[lab].append((ts, pred))

        elif lab in NON_PLAYERS:
            nonplayers += 1

            if pred is not None:
                false_assignments += 1

    # Count identity rows without corresponding raw pose rows.
    missing_identity_rows = sum(
        1 for key in identity_by_key if key not in raw_by_key
    )

    # Evaluate timestamps where both players have ground-truth labels.
    eligible_timestamps = {
        ts: d for ts, d in by_timestamp.items()
        if "A" in d and "B" in d
    }

    correct_timestamps = sum(
        d["A"] == PLAYER_IDS["A"] and
        d["B"] == PLAYER_IDS["B"]
        for d in eligible_timestamps.values()
    )

    # Count identity switches chronologically.
    switches = 0
    switch_details = []

    for person in ("A", "B"):
        previous = None
        previous_timestamp = None

        for ts, pred in sorted(observations[person]):
            if pred is None:
                continue

            if previous is not None and pred != previous:
                switches += 1

                switch_details.append({
                    "person": person,
                    "from_id": previous,
                    "to_id": pred,
                    "previous_timestamp": previous_timestamp,
                    "timestamp": ts
                })

            previous = pred
            previous_timestamp = ts

    # Calculate recording duration and FPS from unique raw-pose timestamps.
    timestamps = sorted({
        key[0] for key in raw_by_key
    })

    duration = (
        timestamps[-1] - timestamps[0]
        if len(timestamps) >= 2 else None
    )

    switches_min = (
        switches * 60 / duration
        if duration and duration > 0 else None
    )

    timestamp_fps = (
        (len(timestamps) - 1) / duration
        if duration and duration > 0 else None
    )

    both_seen = all(counts[p]["labelled"] > 0 for p in PLAYERS)

    all_correct = all(
        counts[p]["correct"] == counts[p]["labelled"]
        for p in PLAYERS
    )

    success = bool(both_seen and all_correct and switches == 0)

    result = {
        "trial_name": trial_name or Path(identified_poses_csv).stem,
        "condition": condition,
        "crossing_trial": crossing_trial,

        "identity": {
            "evaluated_player_poses": player_poses,
            "correct_player_poses": correct_poses,
            "identity_accuracy_percent": percent(
                correct_poses, player_poses
            ),
            "player_assignment_rate_percent": percent(
                assigned_poses, player_poses
            ),

            "timestamps_with_both_players_labelled": len(
                eligible_timestamps
            ),
            "timestamps_with_both_correct_ids": correct_timestamps,
            "timestamp_identity_accuracy_percent": percent(
                correct_timestamps, len(eligible_timestamps)
            ),

            "per_player": {
                p: {
                    **counts[p],
                    "accuracy_percent": percent(
                        counts[p]["correct"], counts[p]["labelled"]
                    ),
                    "assignment_rate_percent": percent(
                        counts[p]["assigned"], counts[p]["labelled"]
                    )
                }
                for p in ("A", "B")
            },

            "labelled_non_player_poses": nonplayers,
            "false_player_assignments": false_assignments,
            "false_assignment_rate_percent": percent(
                false_assignments, nonplayers
            ),

            "missing_raw_pose_rows": missing_raw_rows,
            "missing_identity_rows": missing_identity_rows,
            "unknown_or_unreviewed_poses": unknown
        },

        "identity_switches": {
            "count": switches,
            "switches_per_minute": switches_min,
            "duration_seconds": duration,
            "details": switch_details
        },

        "trial_success": {
            "success": success,
            "success_percent_for_this_trial": (
                100.0 if success else 0.0
            ),
            "definition": (
                "Both players have labelled poses, every labelled player "
                "pose has the expected ID, and there are no counted ID "
                "switches. Ground truth must cover the complete trial."
            )
        },

        "reentry_recovery": reentry_metrics(
            identity_by_key, reentry_events_csv
        ),

        "recording": {
            "raw_pose_rows": len(raw_rows),
            "identity_rows": len(identity_rows),
            "unique_timestamps": len(timestamps),
            "duration_seconds": duration,
            "fps_from_timestamps": timestamp_fps
        },

        "latency": {
            "mean_processing_ms": None,
            "median_processing_ms": None,
            "estimated_fps_from_processing_time": None,
            "note": (
                "Processing time is not currently recorded in either CSV."
            )
        },

        "spatial_error": {
            "note": (
                "Position error is not calculated because the current CSV "
                "files contain 2D pose joints, not estimated and measured "
                "3D positions."
            )
        }
    }

    if output_json:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)

        out.write_text(
            json.dumps(result, indent=2, allow_nan=False),
            encoding="utf-8"
        )

    return result


def summarize_trials(results):
    """Aggregate results and compare experimental conditions."""
    results = list(results)
    groups = defaultdict(list)

    for result in results:
        groups[str(result.get("condition") or "unspecified")].append(
            result
        )

    def group_summary(items):
        accuracies = [
            r["identity"]["identity_accuracy_percent"]
            for r in items
            if r["identity"]["identity_accuracy_percent"] is not None
        ]

        timestamp_accuracies = [
            r["identity"]["timestamp_identity_accuracy_percent"]
            for r in items
            if r["identity"]["timestamp_identity_accuracy_percent"] is not None
        ]

        switch_rates = [
            r["identity_switches"]["switches_per_minute"]
            for r in items
            if r["identity_switches"]["switches_per_minute"] is not None
        ]

        return {
            "trial_count": len(items),
            "mean_identity_accuracy_percent": mean(accuracies),
            "mean_timestamp_identity_accuracy_percent": mean(
                timestamp_accuracies
            ),
            "total_identity_switches": sum(
                r["identity_switches"]["count"] for r in items
            ),
            "mean_switches_per_minute": mean(switch_rates),
            "trial_success_rate_percent": percent(
                sum(r["trial_success"]["success"] for r in items),
                len(items)
            )
        }

    return {
        "trial_count": len(results),
        "overall": group_summary(results),
        "by_lighting_condition": {
            condition: group_summary(items)
            for condition, items in sorted(groups.items())
        },
        "note": (
            "Use consistent test procedures and multiple trials per "
            "lighting condition. Lighting comparisons are descriptive."
        )
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate two-player pose identity tracking."
    )

    parser.add_argument("raw_poses_csv")
    parser.add_argument("identified_poses_csv")
    parser.add_argument("--name", dest="trial_name")
    parser.add_argument("--condition")
    parser.add_argument("--crossing-trial", action="store_true")
    parser.add_argument("--reentry-events")
    parser.add_argument("--output", default="evaluation_result.json")

    args = parser.parse_args()

    result = evaluate_trial(
        args.raw_poses_csv,
        args.identified_poses_csv,
        trial_name=args.trial_name,
        condition=args.condition,
        crossing_trial=args.crossing_trial,
        reentry_events_csv=args.reentry_events,
        output_json=args.output
    )

    print(json.dumps(result, indent=2, allow_nan=False))
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
