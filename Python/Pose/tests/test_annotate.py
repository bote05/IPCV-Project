import json
import sys

import cv2
import numpy as np
import pytest

from Pose.tools import annotate


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for _ in range(3):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    return path


def session(monkeypatch, video, output, inputs):
    """Run the annotator with scripted input: (x, y) clicks the image, a string presses a key."""
    inputs = list(inputs)
    callbacks = {}

    def wait_key(_delay):
        event = inputs.pop(0)
        if isinstance(event, tuple):
            callbacks[annotate.WINDOW](cv2.EVENT_LBUTTONDOWN, *event, 0, None)
            return 255
        return 13 if event == "enter" else ord(event)

    monkeypatch.setattr(cv2, "namedWindow", lambda *_: None)
    monkeypatch.setattr(cv2, "imshow", lambda *_: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda window, callback: callbacks.__setitem__(window, callback))
    monkeypatch.setattr(cv2, "waitKey", wait_key)
    monkeypatch.setattr(sys, "argv", ["annotate", str(video), str(output), "--frames", "1", "--joints", "nose", "left_shoulder"])
    annotate.main()
    assert not inputs
    return json.loads(output.read_text(encoding="utf-8"))


def test_quitting_keeps_an_unfinished_frame_and_resumes_it(monkeypatch, video, tmp_path):
    output = tmp_path / "gt.json"
    data = session(monkeypatch, video, output, [(10, 20), "q"])
    assert data["frames"] == {}
    assert data["partial"] == {"0": [{"nose": [10, 20]}]}

    data = session(monkeypatch, video, output, [(30, 40), "enter"])
    assert data["frames"] == {"0": [{"nose": [10, 20], "left_shoulder": [30, 40]}]}
    assert "partial" not in data


def test_quitting_an_untouched_frame_saves_nothing_for_it(monkeypatch, video, tmp_path):
    data = session(monkeypatch, video, tmp_path / "gt.json", ["q"])
    assert data["frames"] == {}
    assert "partial" not in data
