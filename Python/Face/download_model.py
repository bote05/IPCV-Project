"""Download the OpenCV YuNet face detector (2023mar) and SFace face recognizer (2021dec) into Models/.

    python -m Face.download_model        # run from the Python/ folder
"""

from __future__ import annotations

import urllib.request

from .faceTracking import MODEL_PATH, MODELS_DIR, RECOGNIZER_PATH

ZOO = 'https://github.com/opencv/opencv_zoo/raw/main/models/'
DOWNLOADS = {
    MODEL_PATH: ZOO + 'face_detection_yunet/face_detection_yunet_2023mar.onnx',
    RECOGNIZER_PATH: ZOO + 'face_recognition_sface/face_recognition_sface_2021dec.onnx',
}


def download() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for path, url in DOWNLOADS.items():
        if path.is_file():
            print(f'{path.name} already present')
            continue
        partial = path.with_suffix('.part')
        print(f'downloading {path.name} ...')
        urllib.request.urlretrieve(url, partial)
        partial.replace(path)


if __name__ == '__main__':
    download()
