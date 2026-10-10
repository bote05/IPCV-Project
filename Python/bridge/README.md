# Bridge

Sends the tracking data from Python to Unity over UDP on localhost.

Use Python 3.12 for tracking and Unity 6000.4.8f1. The package-free demo also works on Python 3.10+.

- Port 5005: tracking data as JSON, 30 times per second
- Port 5006: face crops as 256x256 JPEGs, 10 times per second

## Demo

You don't need a webcam for this. It sends two fake players.

```powershell
python Python/main.py --demo
```

In Unity, make an empty GameObject, add `UdpFrameReceiver`, `UdpFaceReceiver`, `BridgeMetrics` and `BridgeDebugView`, and press Play. The panel in the top left shows both players. If you stop Python the players disappear.

## Adding your module

`main.py` opens the webcam and calls `process_frame(frame_bgr, captured_time_s)` for every frame. The image is not mirrored. The timestamp is monotonic seconds, taken just after the camera returns the frame; pass that same timestamp to Pose.

```powershell
python Python/main.py --webcam --processor team_pipeline:process_frame
```

```python
from Pose import PoseDetector, PoseTracker
from bridge.pipeline import TrackingResult, player_from_pose

detector = PoseDetector(num_poses=2)
tracker = PoseTracker()

def process_frame(frame_bgr, captured_time_s):
    poses = detector.detect(frame_bgr, captured_time_s)
    ids = identity(poses)  # Noah's callable: one id (1, 2, or None) per detection
    players = tracker.update(poses, captured_time_s, ids=ids)
    size = (frame_bgr.shape[1], frame_bgr.shape[0])
    return TrackingResult(players=tuple(
        player_from_pose(p.to_dict(size, mirror=True)) for p in players
    ))
```

This is the integration outline after Pose is merged. The combined `team_pipeline` module and `identity` callable still need wiring with Noah and Jona. Keep the detector and tracker alive between calls: Identity runs between detection and tracking. Use `None` for a spectator; do not turn automatic track IDs into player IDs by adding one. Close the detector when the application shuts down.

`player_from_pose` reads the existing `to_dict()` output, so it does not normalize or mirror a second time. It sends joints, held reaches and events; Pose's body-relative diagnostics stay in Pose. Add Jona's face fields/crops and Noah's metre position to the result when their adapters are ready. Unknown fields stay empty. If the processor crashes, Unity gets no players until it recovers. A broken face JPEG is just skipped.

The shared packages are in `Python/requirements.txt`:

```powershell
python -m pip install -r Python/requirements.txt
```

Use a fresh Python 3.12 virtual environment. In an existing environment that has `opencv-python`, uninstall it and reinstall the shared requirements with `--force-reinstall` so `opencv-contrib-python` owns `cv2`. Do not install both OpenCV packages. Alexandra's duplicate Pose requirements can be removed from her branch once she adopts the shared file.

## What gets sent

| Field | What it is |
|---|---|
| `session_id`, `sequence` | New id every time Python starts, and a frame counter. Unity ignores old or duplicate packets. |
| `captured_time_s`, `sent_time_s`, `processing_ms` | When the frame was captured and sent, used for timing |
| `id`, `tracked` | Player 1 or 2, from Identity |
| `pose` | Empty, or 13 joints with `position: [x, y]` and `state`. Coordinates are normalized to image size, origin top left. No depth or invented confidence. |
| `face_bbox` | Empty, or x, y, width, height normalized to image size; match the pose's display mirroring when integrating Face |
| `head_rotation_deg` | Empty, or yaw, pitch, roll in degrees |
| `world_position_m` | Empty, or x, y, z in metres |
| `motion_signals` | Held actions with `name` and `active`: `left_reach`, `right_reach` |
| `events` | Recent pull/jump events with `id`, `type`, `side`, `strength`, `time`; repeated by Pose for 200 ms |

Joint order matches `Pose.Joint` and the bridge's `Joint` / Unity `PoseJoint`: nose, left/right shoulders, elbows, wrists, hips, knees, ankles. States are 0 lost, 1 acquiring, 2 predicted, 3 tracked. Only states 2 and 3 have usable positions; normalized coordinates can extend outside the image.

Empty means unknown, so don't send 0 for missing data. The pose adapter keeps short-gap predictions while Pose's body frame is valid, even when `detected` is false. It sends `tracked=False` with empty data when `valid` becomes false. A lost wrist turns its reach off through Pose's `hand_raised` output. Omitted players also count as lost.

Event IDs increase for the lifetime of the tracker. Keep one tracker for each Python sender session. Unity deduplicates per player and session, including across temporary player loss. The 200 ms repetition reduces loss; a longer outage can still miss an event. Strength is stroke distance for pulls or upward speed for jumps, not confidence. Game cooldowns belong in Unity.

A face packet is the text `IPCF`, the session id, the player id, a counter and then the JPEG bytes.

## Coordinates and remaining team agreements

1. Mirroring: detection and identity use the normal camera image. `to_dict(size, mirror=True)` prepares the pose for Unity's mirrored display; do not flip it again in Unity. Names remain the player's own left/right. A face box from the normal image needs `x = 1 - x - width` for that same display.
2. Position: metres from the camera, x to the right, y down, z away from the camera, measured at the middle of the hips.
3. Head rotation: 0 means looking straight at the camera.
4. Actions: `hand_raised` maps to held reaches; pull/jump use the event callback below. Badr decides their gameplay effects and cooldowns.
5. Losing a player: let go of everything they were holding and hide their effects.

## Using it in Unity

```csharp
if (receiver.TryGetPlayer(1, out PlayerState p) && p.TryGetSignal("left_reach", out MotionSignal s) && s.active) { /* grab */ }
if (receiver.TryGetPlayer(1, out p) && p.TryGetJoint(PoseJoint.LeftWrist, out PoseLandmark wrist)) { /* use wrist.position */ }
if (faces.TryGetTexture(1, out Texture2D face)) { /* show face */ } else { /* hide it */ }
```

Call these in `Update`. `TryGetJoint` returns only tracked/predicted positions. Clear held controls when the player or reach is absent/inactive. Don't change the data you get back.

Subscribe to `receiver.MotionReceived` in `OnEnable` and unsubscribe in `OnDisable`:

```csharp
void OnEnable() => receiver.MotionReceived += HandleMotion;
void OnDisable() => receiver.MotionReceived -= HandleMotion;

void HandleMotion(int playerId, MotionEvent motion)
{
    if (motion.type == "pull") { /* pull with motion.side */ }
    else if (motion.type == "jump") { /* jump */ }
}
```

The callback runs on Unity's main thread once per event ID per player/session. Use it for actions; the raw `p.events` list deliberately repeats across snapshots. IDs restart safely when Python starts a new sender session.

## Timing and tests

Python and Unity use the same Windows clock, so we can measure how long a frame takes from the camera to Unity. This does not include the camera's own delay or drawing the frame on screen. Press "Start metrics CSV" in the debug panel to save the timings.

The original 33-joint synthetic run averaged 0.6 ms capture to bridge acceptance ([CSV](evaluation/synthetic-baseline.csv)). This is a historical headless transport baseline; real tracking and rendered gameplay still need measuring.

```powershell
python -m unittest discover -s Python/bridge/tests -t Python
powershell -ExecutionPolicy Bypass -File Python/bridge/tests/run_unity_tests.ps1 -Editor "<path>/Unity.exe"
```

Three Python integration tests also exercise the real Pose tracker: IDs/mirroring, prediction/loss, and repeated pull events. They require the shared packages and are explicitly skipped when Pose's code is absent. Before merging Pose, its exported `Python/` directory can be added to `PYTHONPATH` for these tests.
