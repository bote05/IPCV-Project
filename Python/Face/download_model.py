"""Download the MediaPipe face detector (BlazeFace short range) into Models/.

    python -m Face.download_model        # run from the Python/ folder
"""

from __future__ import annotations

import urllib.request

from .faceTracking import MODEL_PATH, MODELS_DIR

URL = ('https://storage.googleapis.com/mediapipe-models/face_detector/'
       'blaze_face_short_range/float16/latest/blaze_face_short_range.tflite')


def download() -> None:
    if MODEL_PATH.is_file():
        print(f'{MODEL_PATH.name} already present')
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    partial = MODEL_PATH.with_suffix('.part')
    print(f'downloading {MODEL_PATH.name} ...')
    urllib.request.urlretrieve(URL, partial)
    partial.replace(MODEL_PATH)


if __name__ == '__main__':
    download()
