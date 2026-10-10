"""The format of the tracking data sent to Unity."""

from dataclasses import asdict, dataclass
import json
import math
import string
from typing import Iterable

MAX_PACKET_BYTES = 16_384


@dataclass(frozen=True)
class Landmark:
    position: tuple[float, float, float]
    visibility: float


@dataclass(frozen=True)
class MotionSignal:
    name: str
    active: bool
    confidence: float


@dataclass(frozen=True)
class PlayerState:
    id: int
    tracked: bool
    pose: tuple[Landmark, ...] = ()
    face_bbox: tuple[float, ...] = ()
    head_rotation_deg: tuple[float, ...] = ()
    world_position_m: tuple[float, ...] = ()
    motion_signals: tuple[MotionSignal, ...] = ()


def pose_from_mediapipe(landmarks: Iterable) -> tuple[Landmark, ...]:
    """Converts MediaPipe landmarks, keeping all 33 in order."""
    return tuple(Landmark((p.x, p.y, p.z), p.visibility) for p in landmarks)


def valid_session_id(value: str) -> bool:
    return isinstance(value, str) and len(value) == 32 and all(c in string.hexdigits for c in value)


def finite_vector(values, length: int) -> bool:
    try:
        return len(values) == length and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in values
        )
    except OverflowError:
        return False


def validate_players(players: tuple[PlayerState, ...]) -> None:
    if len(players) > 2:
        raise ValueError("At most two players are supported")
    ids = set()
    for p in players:
        if type(p.id) is not int or p.id not in (1, 2) or p.id in ids:
            raise ValueError("Player IDs must be unique integers: 1 or 2")
        ids.add(p.id)
        if type(p.tracked) is not bool:
            raise ValueError("tracked must be a boolean")
        if not p.tracked and any((p.pose, p.face_bbox, p.head_rotation_deg, p.world_position_m, p.motion_signals)):
            raise ValueError("Untracked players must have empty tracking data")
        if len(p.pose) not in (0, 33):
            raise ValueError("pose must contain zero or 33 landmarks")
        for point in p.pose:
            if not finite_vector(point.position, 3) or not finite_vector((point.visibility,), 1) or not 0 <= point.visibility <= 1:
                raise ValueError("Landmarks need three finite coordinates and visibility 0..1")
        for name, values, size in (
            ("face_bbox", p.face_bbox, 4),
            ("head_rotation_deg", p.head_rotation_deg, 3),
            ("world_position_m", p.world_position_m, 3),
        ):
            if values and not finite_vector(values, size):
                raise ValueError(f"{name} must be empty or contain {size} finite numbers")
        if p.face_bbox and (p.face_bbox[2] < 0 or p.face_bbox[3] < 0):
            raise ValueError("Face width/height cannot be negative")
        names = set()
        for signal in p.motion_signals:
            if not isinstance(signal.name, str) or not signal.name or signal.name in names:
                raise ValueError("Motion signal names must be nonempty and unique per player")
            names.add(signal.name)
            if type(signal.active) is not bool or not finite_vector((signal.confidence,), 1) or not 0 <= signal.confidence <= 1:
                raise ValueError("Motion signals need boolean active and confidence 0..1")


def encode_frame(players: Iterable[PlayerState], *, session_id: str, sequence: int,
                 captured_time_s: float, sent_time_s: float, clock: str) -> bytes:
    players = tuple(players)
    validate_players(players)
    if not valid_session_id(session_id):
        raise ValueError("session_id must be a UUID's 32-character hex string")
    if type(sequence) is not int or not 0 <= sequence <= 2**63 - 1:
        raise ValueError("sequence must fit a nonnegative C# long")
    if not finite_vector((captured_time_s, sent_time_s), 2) or not 0 < captured_time_s <= sent_time_s:
        raise ValueError("Capture/send times must be positive monotonic seconds in order")
    if clock not in ("qpc", "local"):
        raise ValueError("clock must be qpc (Windows) or local")
    packet = json.dumps({
        "session_id": session_id, "sequence": sequence,
        "clock": clock, "captured_time_s": captured_time_s, "sent_time_s": sent_time_s,
        "processing_ms": (sent_time_s - captured_time_s) * 1000,
        "players": [asdict(p) for p in players],
    }, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(packet) > MAX_PACKET_BYTES:
        raise ValueError(f"Tracking snapshot exceeds {MAX_PACKET_BYTES} bytes")
    return packet
