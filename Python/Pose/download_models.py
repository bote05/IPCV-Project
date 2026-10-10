"""Download the MediaPipe Pose Landmarker models into Models/.

    python -m Pose.download_models            # full only
    python -m Pose.download_models lite heavy
"""

from __future__ import annotations

import argparse
import urllib.request

from .detector import MODEL_VARIANTS, MODELS_DIR, model_path

URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{0}/float16/1/pose_landmarker_{0}.task"


def download(variant: str) -> None:
    target = model_path(variant)
    if target.is_file():
        print(f"{target.name} already present")
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    print(f"downloading {target.name} ...")
    urllib.request.urlretrieve(URL.format(variant), partial)
    partial.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="*", choices=MODEL_VARIANTS, metavar="VARIANT", help=f"any of {MODEL_VARIANTS}")
    for variant in parser.parse_args().variants or ["full"]:
        download(variant)


if __name__ == "__main__":
    main()
