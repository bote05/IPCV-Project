"""Live pose tracking demo and session recorder.

    python -m Pose.tools.demo                                  # webcam 0
    python -m Pose.tools.demo --source clip.mp4                # video file, every frame processed
    python -m Pose.tools.demo --record recordings/s1.jsonl --save-video recordings/s1.mp4

Keys: q quit, r toggle raw detections, o simulate arm occlusion, d simulate detector dropout.
"""

from __future__ import annotations

import argparse
import threading
import time
from collections import deque
from dataclasses import replace

import cv2
import numpy as np

from ..detector import MODEL_VARIANTS, PoseDetector, RawPose
from ..evaluation.recording import FrameRecord, RecordingWriter
from ..skeleton import ELBOWS, WRISTS, Joint
from ..tracker import PlayerPose, PoseTracker
from .visualize import EventFlash, SignalPlot, draw_player, draw_raw, draw_text, player_color

OCCLUDED_JOINTS = [*ELBOWS, *WRISTS]
PLOTTED_JOINT = Joint.RIGHT_WRIST


class CameraStream:
    """Grabs camera frames on a background thread and hands out only the newest one. Otherwise
    frames queue up in the driver whenever processing is slower than the camera and the game
    reacts to increasingly old images."""

    def __init__(self, index: int, width: int, height: int):
        self._capture = cv2.VideoCapture(index)
        if not self._capture.isOpened():
            raise RuntimeError(f"cannot open camera {index}")
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.fps = self._capture.get(cv2.CAP_PROP_FPS) or 30.0
        self._condition = threading.Condition()
        self._latest: tuple[np.ndarray, float] | None = None
        self._running = True
        self._thread = threading.Thread(target=self._grab, daemon=True)
        self._thread.start()

    def _grab(self) -> None:
        while self._running:
            ok, frame = self._capture.read()
            with self._condition:
                if ok:
                    self._latest = (frame, time.perf_counter())
                else:
                    self._running = False
                self._condition.notify()

    def read(self) -> tuple[np.ndarray, float] | None:
        with self._condition:
            self._condition.wait_for(lambda: self._latest is not None or not self._running)
            latest, self._latest = self._latest, None
            return latest

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self._capture.release()


class VideoFileStream:
    """Reads every frame of a video; time is the frame's position in the video, so results are
    reproducible regardless of how fast the machine processes them."""

    def __init__(self, path: str):
        self._capture = cv2.VideoCapture(path)
        if not self._capture.isOpened():
            raise RuntimeError(f"cannot open video {path}")
        self.fps = self._capture.get(cv2.CAP_PROP_FPS) or 30.0
        self._index = 0

    def read(self) -> tuple[np.ndarray, float] | None:
        ok, frame = self._capture.read()
        if not ok:
            return None
        t = self._index / self.fps
        self._index += 1
        return frame, t

    def close(self) -> None:
        self._capture.release()


def degrade(frame: np.ndarray, brightness: float, noise: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate low light: scale intensities and add sensor noise (std in grey levels)."""
    if brightness == 1.0 and noise == 0.0:
        return frame
    out = frame.astype(np.float32) * brightness
    if noise > 0:
        out += rng.normal(0.0, noise, frame.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def _flip_x(points: np.ndarray, width: int) -> np.ndarray:
    flipped = points.copy()
    flipped[..., 0] = width - flipped[..., 0]
    return flipped


def mirror_for_display(poses: list[RawPose], players: list[PlayerPose], width: int) -> tuple[list[RawPose], list[PlayerPose]]:
    """Pose copies to draw on a horizontally flipped view; tracking itself never sees a flipped image."""
    poses = [RawPose(_flip_x(pose.points, width), pose.confidence) for pose in poses]
    players = [
        replace(p, position=_flip_x(p.position, width), motion=replace(p.motion, center=_flip_x(p.motion.center, width)))
        for p in players
    ]
    return poses, players


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0", help="camera index or video file")
    parser.add_argument("--model", default="full", help=f"one of {MODEL_VARIANTS} or a .task path")
    parser.add_argument("--players", type=int, default=2, help="maximum number of people to detect")
    parser.add_argument("--width", type=int, default=1280, help="requested camera width")
    parser.add_argument("--height", type=int, default=720, help="requested camera height")
    parser.add_argument("--mirror", action="store_true", help="mirror the display; tracking always uses the camera image as is")
    parser.add_argument("--brightness", type=float, default=1.0, help="intensity scale to simulate low light")
    parser.add_argument("--noise", type=float, default=0.0, help="gaussian noise std added after --brightness")
    parser.add_argument("--record", help="write raw detections to this .jsonl file")
    parser.add_argument("--save-video", help="write the processed frames to this .mp4 file")
    parser.add_argument("--headless", action="store_true", help="no window, e.g. to batch-process a video")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stream = CameraStream(int(args.source), args.width, args.height) if args.source.isdigit() else VideoFileStream(args.source)
    detector = PoseDetector(args.model, num_poses=args.players)
    tracker = PoseTracker()
    rng = np.random.default_rng(0)
    on = tracker.config.filter.confidence_on

    recorder: RecordingWriter | None = None
    video: cv2.VideoWriter | None = None
    flash = EventFlash()
    plot: SignalPlot | None = None
    show_raw, occluding, dropping = True, False, False
    start: float | None = None
    detect_times: list[float] = []
    frame_clock: deque[float] = deque(maxlen=30)
    index = 0

    try:
        while (item := stream.read()) is not None:
            frame, capture_time = item
            start = capture_time if start is None else start
            t = capture_time - start
            frame_clock.append(time.perf_counter())

            frame = degrade(frame, args.brightness, args.noise, rng)
            height, width = frame.shape[:2]

            tic = time.perf_counter()
            poses = detector.detect(frame, t)
            detect_ms = (time.perf_counter() - tic) * 1000
            if dropping:
                poses = []
            elif occluding:
                poses = [pose.hide(OCCLUDED_JOINTS) for pose in poses]

            tic = time.perf_counter()
            players = tracker.update(poses, t)
            track_ms = (time.perf_counter() - tic) * 1000

            if args.record:
                if recorder is None:
                    meta = {"source": args.source, "model": args.model, "brightness": args.brightness, "noise": args.noise}
                    recorder = RecordingWriter(args.record, (width, height), meta)
                brightness = cv2.mean(cv2.cvtColor(frame[::4, ::4], cv2.COLOR_BGR2GRAY))[0]
                recorder.write(FrameRecord(index, t, poses, detect_ms, brightness))
            if args.save_video:
                if video is None:
                    video = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), stream.fps, (width, height))
                video.write(frame)
            index += 1
            detect_times.append(detect_ms)

            if args.headless:
                if index % 100 == 0:
                    print(f"{index} frames, detect {np.mean(detect_times[-100:]):.1f} ms")
                continue

            view = frame.copy()
            shown_poses, shown_players = poses, players
            if args.mirror:
                view = cv2.flip(view, 1)
                shown_poses, shown_players = mirror_for_display(poses, players, width)
            if show_raw:
                for pose in shown_poses:
                    draw_raw(view, pose, on)
            for player in shown_players:
                draw_player(view, player)
                flash.add(player)
            flash.draw(view, t)

            if players:
                tracked = players[0]
                title = f"P{tracked.id} {PLOTTED_JOINT.name.lower()} y"
                if plot is None or plot.title != title:
                    plot = SignalPlot(title)
                raw_y = np.nan
                if tracked.raw is not None and tracked.raw.confidence[PLOTTED_JOINT] >= on:
                    raw_y = tracked.raw.points[PLOTTED_JOINT, 1]
                plot.add(t, raw_y, tracked.position[PLOTTED_JOINT, 1], int(tracked.state[PLOTTED_JOINT]))
                plot.draw(view, (10, height - 130, min(420, width - 20), 120), player_color(tracked.id))

            fps = (len(frame_clock) - 1) / max(frame_clock[-1] - frame_clock[0], 1e-6)
            hud = [
                f"{fps:.0f} fps  detect {detect_ms:.1f} ms  track {track_ms:.2f} ms  model {args.model}",
                f"[r] raw {'on' if show_raw else 'off'}  [o] arm occlusion {'ON' if occluding else 'off'}  "
                f"[d] detector dropout {'ON' if dropping else 'off'}",
            ]
            for row, line in enumerate(hud):
                draw_text(view, line, (10, 22 + 20 * row))
            cv2.imshow("Pose tracking", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            show_raw ^= key == ord("r")
            occluding ^= key == ord("o")
            dropping ^= key == ord("d")
    finally:
        stream.close()
        detector.close()
        if recorder:
            recorder.close()
        if video:
            video.release()
        cv2.destroyAllWindows()

    if detect_times:
        print(f"{index} frames, detection {np.mean(detect_times):.1f} ms mean / {np.percentile(detect_times, 95):.1f} ms p95")


if __name__ == "__main__":
    main()
