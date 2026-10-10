"""What the per-frame tracking function returns."""

from dataclasses import dataclass, field
from .protocol import PlayerState


@dataclass
class TrackingResult:
    players: tuple[PlayerState, ...] = ()
    face_jpegs: dict[int, bytes] = field(default_factory=dict)
