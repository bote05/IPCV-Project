# Bridge

Sends the tracking data from Python to Unity over UDP on localhost.

Use Python 3.10+ and Unity 6000.4.8f1.

- Port 5005: tracking data as JSON, 30 times per second
- Port 5006: face crops as 256x256 JPEGs, 10 times per second

## Demo

You don't need a webcam for this. It sends two fake players.

```powershell
python Python/main.py --demo
```

In Unity, make an empty GameObject, add `UdpFrameReceiver`, `UdpFaceReceiver`, `BridgeMetrics` and `BridgeDebugView`, and press Play. The panel in the top left shows both players. If you stop Python the players disappear.

## Adding your module

`main.py` opens the webcam and calls one function for every frame. That function runs Face, Pose and Identity and returns the result. The frame is not mirrored.

```powershell
python Python/main.py --webcam --processor team_pipeline:process_frame
```

```python
from bridge import PlayerState, MotionSignal
from bridge.pipeline import TrackingResult
from bridge.protocol import pose_from_mediapipe

def process_frame(frame_bgr):
    p1 = PlayerState(id=1, tracked=True,
        pose=pose_from_mediapipe(landmarks),                           # Alexandra
        face_bbox=(x, y, w, h), head_rotation_deg=(yaw, pitch, roll),  # Jona
        world_position_m=(x_m, y_m, z_m),                              # Noah
        motion_signals=(MotionSignal("left_reach", True, 0.9),))       # Alexandra
    return TrackingResult(players=(p1,), face_jpegs={1: jpeg_bytes})   # Jona
```

If you don't have a value yet, leave it out and it stays empty. If the function crashes, Unity gets no players until it works again. A broken face JPEG is just skipped.

Install the webcam packages with `pip install -r Python/bridge/requirements.txt`. Careful when adding MediaPipe: it installs `opencv-contrib-python`, which conflicts with `opencv-python`. Only keep one of them.

## What gets sent

| Field | What it is |
|---|---|
| `session_id`, `sequence` | New id every time Python starts, and a frame counter. Unity ignores old or duplicate packets. |
| `captured_time_s`, `sent_time_s`, `processing_ms` | When the frame was captured and sent, used for timing |
| `id`, `tracked` | Player 1 or 2, from Identity |
| `pose` | Empty, or the 33 MediaPipe landmarks (x, y, z, visibility). x/y are normalized image coordinates; z is relative depth, not metres. |
| `face_bbox` | Empty, or the face box as x, y, width, height (0 to 1, from the top left) |
| `head_rotation_deg` | Empty, or yaw, pitch, roll in degrees |
| `world_position_m` | Empty, or x, y, z in metres |
| `motion_signals` | List of actions with a name, on/off and a confidence |

Empty means we don't know the value, so don't send 0 for that. When a player is lost, leave them out or send `tracked=False` with nothing else.

A face packet is the text `IPCF`, the session id, the player id, a counter and then the JPEG bytes.

## Things we still have to agree on

1. Mirroring: everything uses the normal camera image. Only the preview and the Unity view get flipped.
2. Position: metres from the camera, x to the right, y down, z away from the camera, measured at the middle of the hips.
3. Head rotation: 0 means looking straight at the camera.
4. Actions: `left_reach` and `right_reach` stay on while the arm is up. Cooldowns are done in Unity.
5. Losing a player: let go of everything they were holding and hide their effects.

## Using it in Unity

```csharp
if (receiver.TryGetPlayer(1, out PlayerState p) && p.TryGetSignal("left_reach", out MotionSignal s) && s.active) { /* grab */ }
if (faces.TryGetTexture(1, out Texture2D face)) { /* show face */ } else { /* hide it */ }
```

Call these in `Update`. Don't change the data you get back.

## Timing and tests

Python and Unity use the same Windows clock, so we can measure how long a frame takes from the camera to Unity. This does not include the camera's own delay or drawing the frame on screen. Press "Start metrics CSV" in the debug panel to save the timings.

In the 10-second test with fake data, Unity received about 30 tracking updates per second with no sequence gaps. Capture to bridge acceptance averaged 0.6 ms ([CSV](evaluation/synthetic-baseline.csv)). This was a headless test; real tracking and rendered gameplay still need measuring.

```powershell
python -m unittest discover -s Python/bridge/tests -t Python
powershell -ExecutionPolicy Bypass -File Python/bridge/tests/run_unity_tests.ps1 -Editor "<path>/Unity.exe"
```
