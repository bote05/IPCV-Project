# Task 2: Body pose estimation and motion tracking

Responsible: Alexandra Lupan

Turns webcam frames into smooth, per-player joint positions and climbing control signals, with
controlled behaviour when joints are uncertain or missing.

```
frame ─► PoseDetector ─► RawPose per person ─► PoseTracker ─► PlayerPose per player ─► to_dict() ─► UDP bridge
         MediaPipe Pose     13 joints + confidence    │
         Landmarker         (implausible poses        ├─ association (Hungarian, predicted joints)
         (VIDEO mode)        dropped)                 ├─ KeypointFilter: gating, One Euro, gap prediction
                                                      └─ MotionAnalyzer: body frame, signals, events
```

## Setup

MediaPipe has no wheels for Python 3.14; use Python 3.12. From the repository root:

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r Python/Pose/requirements.txt
cd Python
python -m Pose.download_models          # Models/pose_landmarker_full.task (add lite heavy for comparisons)
python -m pytest Pose/tests
```

All commands below run from `Python/`.

## Demo

```
python -m Pose.demo                                   # webcam 0
python -m Pose.demo --source clip.mp4                 # video file, every frame processed
python -m Pose.demo --mirror --record ../recordings/s1.jsonl --save-video ../recordings/s1.mp4
```

`--mirror` only flips what is shown. Tracking, recordings and saved video always use the camera
image as is, the same as in the integrated game.

Thick skeleton: tracked joints. Thin lines and hollow orange joints: predicted through a gap. Grey
crosses: raw detections. The plot at the bottom shows one wrist's raw height against the filtered
estimate, with orange bands where it was predicted. Raised hands are circled; pull and jump events
flash next to the joint.

Keys: `q` quit, `r` raw detections, `o` simulate arm occlusion (wrists and elbows reported as
undetected), `d` simulate a detector dropout (no detections at all).

Other options: `--model lite|full|heavy`, `--players N`, `--width/--height` (camera),
`--brightness 0.3 --noise 8` (simulated low light), `--headless`.

## Integration

```python
from Pose import PoseDetector, PoseTracker

detector = PoseDetector("full", num_poses=2)
tracker = PoseTracker()

# once per frame; t is the capture time in seconds
poses = detector.detect(frame, t)
ids = identity.assign(poses)                  # Task 3: one player id per pose, None to ignore a pose
players = tracker.update(poses, t, ids)
messages = [p.to_dict((frame.shape[1], frame.shape[0]), mirror=True) for p in players]
```

Detection and tracking are separate calls so player identities can be assigned in between. The
`ids` contract:
- one entry per pose, in the order `detect` returned them
- `None` discards a pose (a spectator, a false detection)
- no id twice in one frame; a mismatch raises `ValueError`
- a player id with no pose for 1 s is dropped; when it comes back, its filters restart cleanly

Without `ids`, the tracker links detections frame to frame itself, which is enough for the demo
and for evaluation. To match poses to players, the identity module can use the previous frame's
`PlayerPose` list, whose joints are smoothed and predicted through gaps.

`to_dict` gives one JSON-ready object per player. Image positions are already normalised to 0..1,
so the bridge does not need the image size. The output uses lists rather than maps so Unity's
`JsonUtility` can parse it, and never contains NaN.

**Mirroring:** processing runs on the camera image as is, and `to_dict(..., mirror=True)` flips
the output for a mirrored display: `x` becomes 1 - x, and body x, horizontal velocities and lean
change sign. Joint names stay the player's own left and right, which on a mirrored display appear
on the same side of the screen as for the player.

**Events** are repeated in every message for 200 ms (`TrackerConfig.event_hold`), so a dropped
UDP message cannot lose one. Each event has an `id` that is unique for the session, so Unity acts
on each id once and needs no cooldown to avoid double counting.

| field | meaning |
|---|---|
| `id`, `time`, `detected` | track id, capture time, whether the player was detected in this frame |
| `valid` | body frame available (both shoulders tracked or predicted) |
| `scale`, `center` | torso length / frame height; hip midpoint, normalised image coordinates |
| `center_velocity` | hip velocity, torso lengths/s, y up |
| `lean` | shoulder roll in degrees, counter-clockwise on screen |
| `arm_extension` | [left, right], 1 = straight arm |
| `hand_raised` | [left, right], wrist above head height |
| `joints[]` | `name`, `state` (0 lost, 1 acquiring, 2 predicted, 3 tracked), `x`/`y` normalised image position, `body` position and `velocity` in the body frame, `moving` |
| `events[]` | `id` (unique per session), `type` (`pull`, `jump`), `side`, `strength`, `time` it fired |

The game should only use joints with `state` 2 or 3.

## Design

**Selected joints.** Nose, shoulders, elbows, wrists, hips, knees and ankles (13 of the 33
BlazePose landmarks). Climbing needs the hands and arms, the torso as a reference frame, and the
legs and hips for jumps. Left and right are the player's own sides: since the image is never
flipped before detection, MediaPipe's labels match the player's anatomy.

**Detector.** MediaPipe Pose Landmarker in VIDEO mode, which tracks people between frames instead
of running the person detector every frame. The `full` model is the default (development laptop
CPU, 768x432 video, `num_poses=2`: lite 16 ms, full 21 ms, heavy 49 ms per frame). Joint
confidence is min(visibility, presence). The landmarker sometimes fits a collapsed skeleton onto
background clutter with high confidence, so poses with a torso under 5% of the frame height are
dropped. Real players are at least ~15%.

**Per-joint state machine** (`filtering.py`): LOST → ACQUIRING → TRACKED ⇄ PREDICTED → LOST.
- Confidence hysteresis: a joint needs 0.6 to be picked up and is dropped below 0.4, so it does
  not flicker around a single threshold.
- Gating: a detection implying more than 15 torso lengths/s, plus a 0.3 torso margin, is rejected
  as a glitch (e.g. a left/right swap or a jump to the other player).
- Gaps: a joint without an accepted detection keeps moving with its last velocity, decaying with
  a 0.1 s time constant, so the extrapolated distance is bounded by v·0.1 s. After 0.4 s it is LOST.
- Re-acquisition: a lost joint must be seen 3 consecutive frames before it is used again, and the
  filter restarts at the new detection instead of sliding there from the old position.

**Smoothing.** One Euro filter (Casiez et al., CHI 2012), vectorised over all joints. At rest the
cutoff is `min_cutoff` = 1 Hz (strong jitter suppression). It rises by `beta` = 1 Hz per torso
length/s of speed, so lag stays small during fast movement. Speed is measured in torso lengths, so
the same parameters behave the same at any distance from the camera. Velocities are the derivative
of the filtered positions, low-passed at 4 Hz. The defaults should be confirmed with
`evaluate sweep` on real sessions.

**Control signals** (`motion.py`) are expressed in a body frame: origin at the shoulder midpoint,
y up, unit = torso length. A player's hands give the same values wherever they stand and however
far from the camera, so one set of thresholds works for both players. Continuous signals use
TRACKED and PREDICTED joints and stay continuous through short occlusions. Discrete outputs
(`hand_raised`, `moving`, events) only change on TRACKED joints and reset when a joint is lost, so
an extrapolated joint can never trigger an action.

- `pull`: a downward wrist stroke relative to the shoulders, at least 0.5 torso lengths from the
  wrist's last high point, moving faster than 1 torso length/s. Fires once per stroke, while the
  stroke is still in progress.
- `jump`: hips rising faster than 2 torso lengths/s with the whole torso tracked and the torso
  length roughly constant. Walking away from a camera mounted above hip height also moves the hips
  up in the image, but it shrinks the torso.

## Evaluation

Record sessions with `demo --record` (and `--save-video` for annotation), then evaluate offline.
Every command replays the recorded detections through the tracker, so any setting can be compared
on identical input in a few seconds.

| report item | command |
|---|---|
| keypoint detection rate | `evaluate summary` (column `detected`) |
| keypoint error | `annotate` on sampled frames, then `evaluate accuracy` (pixel error, % torso, PCK@0.2) |
| jitter, still and moving | `evaluate summary`: RMS deviation while standing still; RMS acceleration while moving; raw vs filtered |
| movement-to-signal delay | `evaluate summary`: filter lag and event delay against a zero-phase reference, plus detection time |
| missing-keypoint handling | `evaluate dropout` (hold vs damped vs constant-velocity prediction; error, availability, recovery time, false events); demo keys `o`/`d` |
| module latency and rate | `evaluate summary`: detection and tracking ms (mean, p95), frame rate and 5th percentile |
| lighting | `evaluate summary rec1 rec2 ...` prints a comparison across recordings with their mean brightness |
| smoothness/responsiveness trade-off | `evaluate sweep` (`--plots` draws the jitter-lag curve) |

```
python -m Pose.evaluate summary ../recordings/*.jsonl --plots ../figures --json ../figures/summary.json
python -m Pose.evaluate sweep ../recordings/s1.jsonl --plots ../figures
python -m Pose.evaluate dropout ../recordings/s1.jsonl --joints arms --plots ../figures
python -m Pose.annotate ../recordings/s1.mp4 ../recordings/s1_gt.json --frames 20
python -m Pose.evaluate accuracy ../recordings/s1.jsonl ../recordings/s1_gt.json
```

There is no ground truth during live play. The **reference** is a zero-phase (forward-backward)
Butterworth low-pass of the raw detections: it follows the motion without delay but needs future
frames, so the live filter's distance from it measures the jitter and lag the live filter leaves.
Still periods are detected automatically (every joint slower than 0.15 torso lengths/s for ≥ 1 s)
or given with `--static START-END`.

A protocol that covers the report template: two players, at 1.5 m and 2.5 m, in normal light,
dim light and backlit, three repetitions each. Each take: stand still 5 s, alternate arm pulls,
raise both hands, jump, hide one hand behind the back, cross in front of each other, one player
leaves and re-enters.

## Limitations

- 2D only: reaching towards the camera is not measured; MediaPipe's depth is too noisy to use.
- Under prolonged occlusion a joint is LOST after 0.4 s and the game gets no position for it.
- MediaPipe's left/right labels can swap for a person seen from the side; gating rejects sudden
  swaps but not a consistent mislabel.
- Fast arm movements blur at low frame rates and low light, which lowers confidence, so joints
  drop out exactly when they move fastest.
- Jump detection assumes a roughly level camera; bending down and standing up quickly can trigger it.
- Track association is short-term only. Persistent identities through crossing and re-entry are Task 3.

## Files

| file | contents |
|---|---|
| `skeleton.py` | selected joints, MediaPipe indices, bones |
| `detector.py` | `PoseDetector`, `RawPose` |
| `filtering.py` | `OneEuroFilter`, `KeypointFilter`, `JointState`, `FilterConfig` |
| `motion.py` | `MotionAnalyzer`, body frame, signals, events |
| `tracker.py` | `PoseTracker`, `PlayerPose.to_dict` |
| `recording.py` | session recording format (JSON lines) |
| `metrics.py`, `evaluate.py` | offline evaluation |
| `annotate.py` | ground-truth annotation tool |
| `demo.py`, `visualize.py` | live demo and drawing |
| `download_models.py` | fetches the pinned model files into `Models/` |
| `tests/` | unit tests on synthetic poses |
