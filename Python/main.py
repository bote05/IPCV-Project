"""Run the bridge demo, the real webcam tracking, or the whole game (--play)."""

import argparse
import logging
import math
from pathlib import Path
import statistics
import subprocess
import threading
import time

from bridge.demo import demo_players
from bridge.pipeline import TrackingResult
from bridge.sender import UdpSender

GAME = Path(__file__).resolve().parents[1] / "Build" / "ClimbingGame.exe"


def positive_fps(value: str) -> float:
    fps = float(value)
    if not math.isfinite(fps) or not 0 < fps <= 240:
        raise argparse.ArgumentTypeError("FPS must be greater than zero and at most 240")
    return fps


def run_demo(sender: UdpSender, fps: float, duration: float | None) -> None:
    assets = Path(__file__).parent / "bridge" / "demo_assets"
    faces = {i: (assets / f"player{i}.jpg").read_bytes() for i in (1, 2)}
    start = deadline = time.perf_counter()
    last_faces = 0.0
    while duration is None or time.perf_counter() - start < duration:
        captured = time.perf_counter()
        sender.send(demo_players(captured - start), captured_time_s=captured)
        if captured - last_faces >= 0.1:
            for player_id, jpeg in faces.items():
                sender.send_face(player_id, jpeg)
            last_faces = captured
        deadline = max(deadline + 1 / fps, time.perf_counter())
        time.sleep(max(0, deadline - time.perf_counter()))


def send_faces(sender: UdpSender, result: TrackingResult) -> None:
    """Send the face crops. A bad crop is skipped so the players still get sent."""
    present = {p.id for p in result.players if p.tracked and p.face_bbox}
    for player_id, jpeg in result.face_jpegs.items():
        if player_id in present:
            try:
                sender.send_face(player_id, jpeg)
            except ValueError as error:
                logging.warning("Skipped face crop for player %s: %s", player_id, error)


class LatestFrame:
    """Read in the background, keeping only the newest frame."""

    def __init__(self, camera):
        self.camera = camera
        self.frame = None
        self.captured = 0.0
        self.failed = False
        self.running = True
        self.ready = threading.Condition()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        try:
            while self.running:
                ok, frame = self.camera.read()
                if not ok:
                    return
                with self.ready:
                    # Include waiting for processing in the capture-to-apply latency.
                    self.frame, self.captured = frame, time.perf_counter()
                    self.ready.notify()
        except Exception:
            logging.exception("Webcam reader failed")
        finally:
            with self.ready:
                self.failed = True
                self.ready.notify()
            # Only this thread releases the camera, after read() has returned.
            self.camera.release()

    def read(self):
        """(ok, frame, capture time) of a frame not returned before, waiting for it if needed."""
        with self.ready:
            self.ready.wait_for(lambda: self.frame is not None or self.failed, timeout=2)
            frame, self.frame = self.frame, None
            return frame is not None, frame, self.captured

    def close(self) -> None:
        self.running = False
        self.thread.join(1)


def log_loop_times(times: dict[str, list[float]]) -> None:
    """Report loop timings and overall throughput."""
    for name, values in times.items():
        if len(values) >= 2:
            p95 = statistics.quantiles(values, n=20, method="inclusive")[18]
            logging.info("%-16s median %5.1f ms  p95 %5.1f ms", name, statistics.median(values) * 1000, p95 * 1000)
    if len(times["whole loop"]) >= 2:
        durations = times["whole loop"]
        logging.info("Tracking rate %.1f fps over %d frames", len(durations) / sum(durations), len(durations))


def run_camera(sender: UdpSender, processor, camera_index: int, fps: float,
               draw=None, on_key=None, size: tuple[int, int] | None = None, game=None,
               camera_thread: bool = False) -> None:
    import cv2  # Only needed for the webcam, not the demo.

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open webcam {camera_index}")
    if size:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    frames = LatestFrame(camera) if camera_thread else None
    failed = False
    last_faces = 0.0
    times = {name: [] for name in ("wait for camera", "processing", "sending", "preview window", "whole loop")}
    try:
        while True:
            started = time.perf_counter()
            if frames:
                ok, frame, captured = frames.read()
            else:
                ok, frame = camera.read()
                captured = time.perf_counter()
            if not ok:
                sender.send(())
                raise RuntimeError("Webcam stopped returning frames")
            times["wait for camera"].append(time.perf_counter() - started)
            try:
                step = time.perf_counter()
                result = processor(frame, captured)  # Normal BGR image and capture time in seconds.
                if not isinstance(result, TrackingResult):
                    raise TypeError("Processor must return bridge.pipeline.TrackingResult")
                times["processing"].append(time.perf_counter() - step)
                step = time.perf_counter()
                sender.send(result.players, captured_time_s=captured, image_size=result.image_size)
                if failed:
                    logging.info("Frame processor recovered")
                failed = False
            except Exception:
                sender.send(())  # Send no players so nothing stays stuck on the old pose.
                if not failed:
                    logging.exception("Frame processor failed; sending no players until recovery")
                failed = True
            else:
                if captured - last_faces >= 0.1:
                    send_faces(sender, result)
                    last_faces = captured
                times["sending"].append(time.perf_counter() - step)
            step = time.perf_counter()
            cv2.imshow("Webcam preview (mirrored display only; Q quits)", draw(frame) if draw else cv2.flip(frame, 1))
            key = cv2.waitKey(1) & 0xFF
            times["preview window"].append(time.perf_counter() - step)
            if key in (ord("q"), ord("Q"), 27) or (game is not None and game.poll() is not None):
                break
            if on_key:
                on_key(key)
            time.sleep(max(0, 1 / fps - (time.perf_counter() - started)))
            times["whole loop"].append(time.perf_counter() - started)
    finally:
        if frames:
            frames.close()
        else:
            camera.release()
        cv2.destroyAllWindows()
        log_loop_times(times)


def main() -> None:
    parser = argparse.ArgumentParser(description="Python-to-Unity tracking bridge")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="Synthetic two-player data and placeholder JPEGs")
    mode.add_argument("--webcam", action="store_true", help="Track two players with Face, Pose and Identity")
    mode.add_argument("--play", action="store_true", help="Start the built game and the webcam tracking together")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height")
    parser.add_argument("--calibration-distance", type=float, default=2.0,
                        help="Metres from the camera where players stand when the C countdown ends")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--face-port", type=int, default=5006)
    parser.add_argument("--fps", type=positive_fps, default=30.0)
    parser.add_argument("--camera-thread", action="store_true",
                        help="Read the webcam in the background (may increase throughput and frame age)")
    parser.add_argument("--duration", type=float, help="Demo duration in seconds; omit to run until Ctrl+C")
    args = parser.parse_args()
    if not all(1 <= p <= 65535 for p in (args.port, args.face_port)) or args.port == args.face_port:
        parser.error("Ports must be distinct and in 1..65535")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0 or not args.demo):
        parser.error("--duration must be positive and is only supported with --demo")
    if args.play and not GAME.is_file():
        parser.error(f"The game is not built yet: {GAME} is missing. See README.md, 'Build the game'")
    if args.width <= 0 or args.height <= 0:
        parser.error("Camera width and height must be positive")
    if not math.isfinite(args.calibration_distance) or args.calibration_distance <= 0:
        parser.error("--calibration-distance must be a positive number of metres")
    tracking = game = None
    if not args.demo:
        from tracking import GameTracking  # Loads MediaPipe, so not for the demo.
        try:
            tracking = GameTracking(args.calibration_distance)
        except FileNotFoundError as error:
            parser.error(f"{error}. See README.md for the model downloads")
    logging.basicConfig(level=logging.INFO)
    with UdpSender(port=args.port, face_port=args.face_port) as sender:
        print(f"{'SIMULATED' if args.demo else 'WEBCAM'} tracking -> 127.0.0.1:{args.port}, faces -> {args.face_port}")
        try:
            if args.demo:
                run_demo(sender, args.fps, args.duration)
            else:
                if args.play:
                    game = subprocess.Popen([str(GAME), "-screen-fullscreen", "0",
                                             "-screen-width", "1280", "-screen-height", "720"])
                run_camera(sender, tracking.process, args.camera, args.fps, tracking.draw, tracking.on_key,
                           (args.width, args.height), game, args.camera_thread)
        except KeyboardInterrupt:
            pass
        finally:
            sender.send(())
            if tracking:
                tracking.close()
            if game and game.poll() is None:
                game.terminate()
                game.wait(10)
            print(f"Sent {sender.frames_sent} states, {sender.faces_sent} crops; buffer drops {sender.frames_dropped}.")


if __name__ == "__main__":
    main()
