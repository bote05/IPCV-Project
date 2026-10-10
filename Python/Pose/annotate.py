"""Click ground-truth joint positions on sampled video frames for `evaluate.py accuracy`.

    python -m Pose.annotate recordings/s1.mp4 recordings/s1_gt.json --frames 20

Click each requested joint of a person. Keys: s skip a joint that is not visible, u undo,
n start the next person, enter/space next frame, q save and quit. Existing annotations are kept,
so a session can be continued later.

Left and right are the person's own sides as MediaPipe labels them: for a person facing the
camera in an unmirrored image, their left wrist appears on the right of the image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .skeleton import JOINT_NAMES
from .visualize import PLAYER_COLORS, draw_text

WINDOW = "annotate"
MAX_DISPLAY_WIDTH = 1280


class FrameAnnotator:
    def __init__(self, frame: np.ndarray, joints: list[str], title: str):
        self.frame = frame
        self.joints = joints
        self.title = title
        self.zoom = min(1.0, MAX_DISPLAY_WIDTH / frame.shape[1])
        self.people: list[dict[str, list[float] | None]] = [{}]

    @property
    def next_joint(self) -> str | None:
        current = self.people[-1]
        return next((name for name in self.joints if name not in current), None)

    def on_mouse(self, event: int, x: int, y: int, *_) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and self.next_joint:
            self.people[-1][self.next_joint] = [x / self.zoom, y / self.zoom]

    def run(self) -> list[dict] | None:
        """Annotated people of this frame, or None when the user quits."""
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        while True:
            cv2.imshow(WINDOW, self._render())
            key = cv2.waitKey(20) & 0xFF
            current = self.people[-1]
            if key == ord("s") and self.next_joint:
                current[self.next_joint] = None
            elif key == ord("u"):
                if current:
                    current.pop(next(reversed(current)))
                elif len(self.people) > 1:
                    self.people.pop()
            elif key == ord("n") and current:
                self.people.append({})
            elif key in (13, 32):
                return [person for person in self.people if any(point is not None for point in person.values())]
            elif key == ord("q"):
                return None

    def _render(self) -> np.ndarray:
        view = cv2.resize(self.frame, None, fx=self.zoom, fy=self.zoom)
        for i, person in enumerate(self.people):
            color = PLAYER_COLORS[i % len(PLAYER_COLORS)]
            for point in person.values():
                if point is not None:
                    cv2.circle(view, (int(point[0] * self.zoom), int(point[1] * self.zoom)), 4, color, -1)
        prompt = f"person {len(self.people)}: click {self.next_joint}  [s] not visible" if self.next_joint else "person done"
        draw_text(view, self.title, (10, 22))
        draw_text(view, prompt, (10, 44), (0, 255, 255))
        draw_text(view, "[u] undo  [n] next person  [enter] next frame  [q] save and quit", (10, 66))
        return view


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("output", help="annotation .json file, created or extended")
    parser.add_argument("--frames", type=int, default=20, help="number of evenly spaced frames to annotate")
    parser.add_argument("--joints", nargs="+", choices=JOINT_NAMES, default=list(JOINT_NAMES), metavar="JOINT",
                        help="joints to annotate, default all")
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output = Path(args.output)
    data = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {"video": args.video, "frames": {}}

    cv2.namedWindow(WINDOW)
    indices = np.linspace(0, total - 1, args.frames).round().astype(int)
    for n, index in enumerate(indices):
        if str(index) in data["frames"]:
            continue
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            continue
        people = FrameAnnotator(frame, args.joints, f"frame {index} ({n + 1}/{len(indices)})").run()
        if people is None:
            break
        data["frames"][str(index)] = people
        output.write_text(json.dumps(data, indent=1), encoding="utf-8")
    cv2.destroyAllWindows()
    print(f"{len(data['frames'])} frames annotated in {output}")


if __name__ == "__main__":
    main()
