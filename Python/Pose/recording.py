"""Session recordings: the raw detections and timing of every frame, stored as JSON lines.

Recording the detector output instead of the filtered result means any filter or motion setting can
be evaluated offline on exactly the same input (see evaluate.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .detector import RawPose
from .skeleton import JOINT_NAMES

FORMAT = "pose-recording/1"


@dataclass(frozen=True)
class FrameRecord:
    index: int  # frame number in the source video
    time: float  # capture time in seconds
    poses: list[RawPose]
    detect_ms: float
    brightness: float  # mean grey level (0-255) of the frame the detector saw


@dataclass(frozen=True)
class Recording:
    image_size: tuple[int, int]
    frames: list[FrameRecord]
    meta: dict = field(default_factory=dict)


class RecordingWriter:
    def __init__(self, path: str | Path, image_size: tuple[int, int], meta: dict | None = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "w", encoding="utf-8")
        header = {"format": FORMAT, "image_size": list(image_size), "joints": JOINT_NAMES, "meta": meta or {}}
        self._file.write(json.dumps(header) + "\n")

    def write(self, record: FrameRecord) -> None:
        line = {
            "index": record.index,
            "time": round(record.time, 4),
            "detect_ms": round(record.detect_ms, 2),
            "brightness": round(record.brightness, 1),
            "poses": [
                {"points": np.round(p.points, 2).tolist(), "confidence": np.round(p.confidence, 3).tolist()}
                for p in record.poses
            ],
        }
        self._file.write(json.dumps(line) + "\n")

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> RecordingWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load_recording(path: str | Path) -> Recording:
    with open(path, encoding="utf-8") as file:
        header = json.loads(file.readline())
        if header.get("format") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} file")
        if tuple(header["joints"]) != JOINT_NAMES:
            raise ValueError(f"{path} was recorded with a different joint set")
        frames = []
        for line in file:
            data = json.loads(line)
            poses = [RawPose(np.array(p["points"], dtype=float), np.array(p["confidence"], dtype=float)) for p in data["poses"]]
            frames.append(FrameRecord(data["index"], data["time"], poses, data["detect_ms"], data["brightness"]))
    return Recording(tuple(header["image_size"]), frames, header["meta"])
