"""Packs a face JPEG into one UDP packet."""

import struct
from .protocol import valid_session_id

FACE_HEADER = struct.Struct("!4s32sBQ")
MAX_FACE_PACKET_BYTES = 60_000


def encode_face(session_id: str, player_id: int, sequence: int, jpeg: bytes) -> bytes:
    if not valid_session_id(session_id) or type(player_id) is not int or player_id not in (1, 2):
        raise ValueError("Faces need the sender session and player ID 1 or 2")
    if not isinstance(jpeg, bytes) or not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
        raise ValueError("Face crop must be JPEG bytes")
    if len(jpeg) + FACE_HEADER.size > MAX_FACE_PACKET_BYTES:
        raise ValueError("JPEG exceeds the 60 KB datagram limit; lower JPEG quality")
    return FACE_HEADER.pack(b"IPCF", session_id.encode("ascii"), player_id, sequence) + jpeg
