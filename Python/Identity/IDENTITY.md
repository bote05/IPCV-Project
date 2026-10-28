# Identity module interface

How persistent player identities (Task 3) plug into pose tracking (Task 2).

## Per frame

```python
from Pose import PoseDetector, PoseTracker

detector = PoseDetector("full", num_poses=2)
tracker = PoseTracker()

poses = detector.detect(frame, t)          # t: capture time in seconds
ids = identity.assign(poses, previous)     # one player id per pose
previous = tracker.update(poses, t, ids)   # list[PlayerPose], used for the next frame
```

## Input: what `detect` returns

A list of `RawPose`, at most `num_poses` long. The order is arbitrary and can change from one frame
to the next.

| field | shape | meaning |
|---|---|---|
| `points` | (13, 2) | joint positions in pixels of the camera image (x right, y down, not mirrored) |
| `confidence` | (13,) | 0..1, min of MediaPipe's visibility and presence; below 0.6 the position is a guess |

Joint order is `Pose.JOINT_NAMES` (or the `Pose.Joint` enum): nose, left/right shoulder, elbow,
wrist, hip, knee, ankle. Left and right are the player's own sides.

Skeletons with a torso shorter than 5% of the frame height are already removed; they are false
detections on background objects.

## Output: what to pass as `ids`

- One entry per pose, in the order `detect` returned them.
- Any integers, e.g. 1 and 2. They become `PlayerPose.id` and the `id` in the JSON sent to Unity.
- `None` discards a pose: a spectator, a false detection, or a duplicate. When `ids` are given the
  tracker does not remove duplicate detections of the same person itself.
- The same id may not appear twice in one frame. That, or a list of the wrong length, raises
  `ValueError`.

## What comes back

`tracker.update` returns one `PlayerPose` per player id that had a pose within the last second,
sorted by id.

| field | meaning |
|---|---|
| `id` | the id you assigned |
| `detected` | a pose was assigned to this id in the current frame |
| `position` | (13, 2) smoothed joints in pixels |
| `velocity` | (13, 2) pixels/s |
| `state` | (13,) `JointState`: 3 tracked, 2 predicted through a short gap, 1 acquiring, 0 lost. Only 2 and 3 are usable |
| `motion.center` | hip midpoint in pixels |
| `motion.scale` | torso length in pixels, for comparing distances independently of how far a player stands |
| `raw` | the `RawPose` assigned this frame, or None |

The previous frame's list is the natural thing to match new poses against: its joints are
smoothed, and predicted for up to 0.4 s while hidden. For example:

```python
import numpy as np
from Pose import JointState, PlayerPose, RawPose

def distance(pose: RawPose, player: PlayerPose, dt: float) -> float:
    """Mean joint distance in torso lengths between a new pose and where a player is expected."""
    expected = player.position + player.velocity * dt
    common = (player.state >= JointState.PREDICTED) & (pose.confidence >= 0.6)
    if not common.any():
        return float("inf")
    return float(np.linalg.norm(pose.points[common] - expected[common], axis=1).mean() / max(player.motion.scale, 1.0))
```

## Lifecycle of an id

- Missing for less than 1 s: the player stays in the output with `detected` False. Joints are
  predicted for 0.4 s, then lost.
- Missing for more than 1 s: the player is dropped. When the id is given a pose again, its filters
  start from scratch, with a 3-frame warm-up before joints are usable. A re-entering player
  therefore never slides in from where they left.
- Swapping two ids for even one frame feeds each player's filters the other person's joints, so
  for a moment both skeletons are pulled towards each other. Stable assignments matter more than
  fast ones.

## What the pose side does and does not handle

Handled before or inside the tracker: jitter, joints missing for up to 0.4 s, implausible
skeletons, and single-frame jumps of a joint.

Left to identity:
- which person is which player
- crossing
- one player hiding the other; MediaPipe then often returns a single pose
- leaving and re-entering
- extra people in view: with `num_poses=2` a spectator can take one of the two detection slots, so
  `num_poses=3` plus `None` for the extra pose is safer if spectators are expected

Without `ids`, `tracker.update(poses, t)` links poses frame to frame by itself and numbers tracks
0, 1, 2, and so on. That is enough for the demo and evaluation, but a person who leaves and comes
back gets a new number.
