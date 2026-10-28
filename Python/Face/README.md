# Face

Detects faces with MediaPipe (BlazeFace short range), smooths each box with a One Euro filter and makes a 256x256 JPEG crop per face for Unity.

## Setup

The model isn't in git. Download it once (it goes to `Models/blaze_face_short_range.tflite`), from `Python/`:

```
python -m Face.download_model
```

Uses the shared `Python/requirements.txt`. Needs the `bridge` and `Pose` packages on the import path, which is the case when running from `Python/`.

## Usage

```python
from Face.faceTracking import FaceTracker

face = FaceTracker()                    # FaceTracker(debug=True) shows a debug window
faces = face.process(frame_bgr, t)      # every frame; t = capture time in seconds
face.close()                            # or: with FaceTracker() as face: ...
```

Pass the normal, unmirrored camera frame and the same increasing timestamp the other modules get. Keep one `FaceTracker` alive for the whole run: it holds the detector, the face tracks and the templates.

`process` returns a `list[FaceObservation]`, one per face, in no particular order:

| Field | What it is |
|---|---|
| `track_id` | Same number for the same face across frames. A face that is dropped and comes back gets a new id. **Not a player id.** |
| `bbox` | `(x, y, width, height)` normalized to the image, origin top left, **not mirrored**, smoothed and clipped to the image. For Unity's mirrored display use `x = 1 - x - width`. |
| `jpeg` | 256x256 JPEG of the smoothed box, at most `bridge.face.MAX_FACE_JPEG_BYTES`. Quality starts at 85 and drops until it fits; `None` if it never fits (don't send a face then). |
| `template_scores` | `(score_0, score_1)`, similarity to template 0 and 1 (-1..1, higher is more alike), or `None` until both templates exist. |
| `detected` | `True` if the detector saw the face this frame. `False` while the face is held after a missed detection. |

## Behaviour

- **All faces are returned**, also before the templates exist and also when there are more than two.
- **Templates**: the first frame where exactly two faces are detected, both crops are stored for the rest of the run. Template 0 is the face on the left of the (unmirrored) camera image at that moment, template 1 the one on the right. `reset_templates()` forgets them and captures new ones the next time two faces are seen. The scores are only a hint: this module does not decide who is player 1 or 2.
- **Smoothing**: each face has its own `Pose.filtering.OneEuroFilter` on the box centre and size. Speed is measured in face widths per second. Tune with `FILTER_MIN_CUTOFF` (lower = less jitter) and `FILTER_BETA` (higher = less lag) at the top of `faceTracking.py`.
- **Matching between frames**: a detection continues the track whose smoothed box overlaps it most (IoU at least `MIN_MATCH_IOU = 0.3`); otherwise it starts a new track.
- **Missed detections**: a face that isn't detected is still returned with its last box, JPEG and scores and `detected=False`. After `FACE_TIMEOUT_S = 0.5` seconds without a detection it is dropped, so Unity can hide it.
