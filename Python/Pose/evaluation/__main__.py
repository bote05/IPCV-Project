"""Offline evaluation of the pose module on recorded sessions (record with tools/demo.py --record).

    python -m Pose.evaluation summary recordings/*.jsonl [--static 0-5] [--plots out] [--json out.json]
    python -m Pose.evaluation sweep recordings/s1.jsonl
    python -m Pose.evaluation dropout recordings/s1.jsonl
    python -m Pose.evaluation accuracy recordings/s1.jsonl recordings/s1_gt.json

summary   detection rate, jitter (still and moving), lag, event delay, latency; one table per
          recording plus a comparison across recordings (e.g. lighting conditions)
sweep     jitter/lag trade-off over a grid of One Euro parameters
dropout   hides joints for gaps of increasing length and measures prediction error, recovery time
          and false events, for three gap-handling strategies
accuracy  keypoint error against hand-annotated frames (see tools/annotate.py)

Each command replays the recorded raw detections through the tracker, so every setting is
evaluated on identical input.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ..filtering import FilterConfig
from ..skeleton import JOINT_NAMES, Joint
from ..tracker import TrackerConfig
from . import report
from .experiments import JOINT_GROUPS, accuracy, dropout, summarize, sweep


def _interval(text: str) -> tuple[float, float]:
    start, stop = text.split("-")
    return float(start), float(stop)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m Pose.evaluation", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
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
            report.print_summary(summary)
        if len(summaries) > 1:
            report.print_comparison(summaries)
        if plots:
            joint = Joint[args.joint.upper()]
            for path, (_, items) in zip(args.recordings, results):
                if items:
                    best = max(items, key=lambda a: a.track.confident[:, joint].sum())
                    report.plot_trajectory(best, joint, plots / f"{Path(path).stem}_trajectory.png")
        report.write_json(args.json, summaries)
    elif args.command == "sweep":
        rows = sweep(args.recording, config, args.static, args.reference_cutoff, args.min_cutoffs, args.betas)
        report.print_sweep(rows, config)
        if plots:
            report.plot_sweep(rows, plots / f"{Path(args.recording).stem}_sweep.png")
        report.write_json(args.json, rows)
    elif args.command == "dropout":
        rows = dropout(args.recording, config, JOINT_GROUPS[args.joints], args.durations)
        report.print_dropout(rows, args.joints)
        if plots:
            report.plot_dropout(rows, plots / f"{Path(args.recording).stem}_dropout_{args.joints}.png")
        report.write_json(args.json, rows)
    else:
        result = accuracy(args.recording, args.annotations, config)
        report.print_accuracy(result)
        report.write_json(args.json, result)


if __name__ == "__main__":
    main()
