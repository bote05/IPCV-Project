"""Packs a face JPEG into one UDP packet."""

import struct
from .protocol import MAX_PACKET_BYTES, valid_session_id

FACE_HEADER = struct.Struct("!4s32sBQ")
# Largest JPEG that fits in one packet. Lower the JPEG quality until it is below this.
MAX_FACE_JPEG_BYTES = MAX_PACKET_BYTES - FACE_HEADER.size


def encode_face(session_id: str, player_id: int, sequence: int, jpeg: bytes) -> bytes:
    if not valid_session_id(session_id) or type(player_id) is not int or player_id not in (1, 2):
        raise ValueError("Faces need the sender session and player ID 1 or 2")
    if not isinstance(jpeg, bytes) or not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
        raise ValueError("Face crop must be JPEG bytes")
    if len(jpeg) > MAX_FACE_JPEG_BYTES:
        raise ValueError(f"JPEG is {len(jpeg)} bytes, the limit is {MAX_FACE_JPEG_BYTES}; lower the JPEG quality")
    return FACE_HEADER.pack(b"IPCF", session_id.encode("ascii"), player_id, sequence) + jpeg
