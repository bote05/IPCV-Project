"""Console tables, figures and JSON output for the evaluation results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..filtering import JointState
from ..skeleton import Joint
from ..tracker import TrackerConfig
from .experiments import GAP_STRATEGIES
from .metrics import contiguous_runs
from .replay import MIN_TRACK_DURATION, Analysis


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


def print_sweep(rows: list[dict], config: TrackerConfig) -> None:
    print(f"\n{'min_cutoff':>10}{'beta':>6}{'still jitter %torso':>21}{'moving accel':>14}{'moving error %torso':>21}{'lag ms':>8}")
    for r in rows:
        default = " <- default" if (r["min_cutoff"], r["beta"]) == (config.filter.min_cutoff, config.filter.beta) else ""
        print(f"{r['min_cutoff']:>10.2f}{r['beta']:>6.2f}{100 * r['still_jitter']:>21.2f}{r['moving_acceleration']:>14.1f}"
              f"{100 * r['moving_error']:>21.2f}{r['lag_ms']:>8.0f}{default}")


def print_dropout(rows: list[dict], group: str) -> None:
    print(f"\nhidden joints: {group}")
    print(f"{'strategy':<19}{'gap ms':>7}{'gaps':>6}{'usable':>8}{'median err %torso':>19}{'p90 err':>9}{'recovery ms':>13}{'false events':>14}")
    for r in rows:
        print(f"{r['strategy']:<19}{r['gap_ms']:>7.0f}{r['gaps']:>6}{r['usable']:>8.0%}{100 * r['median_error']:>19.1f}"
              f"{100 * r['p90_error']:>9.1f}{r['recovery_ms']:>13.0f}{r['extra_events']:>14}")


def print_accuracy(result: dict) -> None:
    print(f"\n{result['matched_people']}/{result['annotated_people']} annotated people matched to a detection")
    print(f"{'joint':<15}{'n':>5}{'detected':>10}{'median px':>11}{'mean err %torso':>17}{'filtered':>10}{'PCK@0.2':>9}{'filtered':>10}")
    for name, m in result["joints"].items():
        print(f"{name:<15}{m['annotated']:>5}{m['detection_rate']:>10.0%}{m['median_error_px']:>11.1f}"
              f"{100 * m['mean_error_raw']:>17.1f}{100 * m['mean_error_filtered']:>10.1f}{m['pck_raw']:>9.0%}{m['pck_filtered']:>10.0%}")


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_trajectory(a: Analysis, joint: Joint, path: Path) -> None:
    plt = _pyplot()
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


def plot_sweep(rows: list[dict], path: Path) -> None:
    plt = _pyplot()
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


def plot_dropout(rows: list[dict], path: Path) -> None:
    plt = _pyplot()
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


def write_json(path: str | None, data) -> None:
    if path:
        Path(path).write_text(json.dumps(data, indent=2, default=_to_builtin), encoding="utf-8")


def _to_builtin(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))
