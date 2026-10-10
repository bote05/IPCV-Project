"""Sends tracking data and face crops to Unity over UDP."""

import os
import socket
import time
from typing import Iterable
import uuid

from .face import encode_face
from .protocol import PlayerState, encode_frame


class UdpSender:
    def __init__(self, port: int = 5005, face_port: int = 5006):
        if any(type(p) is not int or not 1 <= p <= 65535 for p in (port, face_port)) or port == face_port:
            raise ValueError("Tracking and face ports must be distinct, in 1..65535")
        self._destination = ("127.0.0.1", port)
        self._face_destination = ("127.0.0.1", face_port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        self.session_id = uuid.uuid4().hex
        self.sequence = 0
        self.face_sequence = 0
        self.frames_sent = self.faces_sent = self.frames_dropped = 0

    def _send(self, packet: bytes, destination) -> bool:
        try:
            self._socket.sendto(packet, destination)
        except BlockingIOError:
            self.frames_dropped += 1
            return False
        return True

    def send(self, players: Iterable[PlayerState], *, captured_time_s: float | None = None) -> bool:
        """captured_time_s should be time.perf_counter() right after reading the camera."""
        sent_time_s = time.perf_counter()
        packet = encode_frame(
            players, session_id=self.session_id, sequence=self.sequence,
            captured_time_s=sent_time_s if captured_time_s is None else captured_time_s,
            sent_time_s=sent_time_s, clock="qpc" if os.name == "nt" else "local",
        )
        self.sequence += 1
        sent = self._send(packet, self._destination)
        self.frames_sent += int(sent)
        return sent

    def send_face(self, player_id: int, jpeg: bytes) -> bool:
        """Send one 256x256 face JPEG for a player."""
        packet = encode_face(self.session_id, player_id, self.face_sequence, jpeg)
        self.face_sequence += 1
        sent = self._send(packet, self._face_destination)
        self.faces_sent += int(sent)
        return sent

    def close(self) -> None:
        self._socket.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
