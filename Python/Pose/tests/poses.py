"""Synthetic poses for the tests: a person standing upright facing the camera."""

import numpy as np

from Pose.detector import RawPose
from Pose.skeleton import NUM_JOINTS, Joint

# Joint offsets from the shoulder midpoint in torso lengths, image orientation (y down). The
# person's left side is on the image's right.
STANDING = {
    Joint.NOSE: (0.0, -0.55),
    Joint.LEFT_SHOULDER: (0.35, 0.0),
    Joint.RIGHT_SHOULDER: (-0.35, 0.0),
    Joint.LEFT_ELBOW: (0.45, 0.45),
    Joint.RIGHT_ELBOW: (-0.45, 0.45),
    Joint.LEFT_WRIST: (0.5, 0.9),
    Joint.RIGHT_WRIST: (-0.5, 0.9),
    Joint.LEFT_HIP: (0.2, 1.0),
    Joint.RIGHT_HIP: (-0.2, 1.0),
    Joint.LEFT_KNEE: (0.2, 1.8),
    Joint.RIGHT_KNEE: (-0.2, 1.8),
    Joint.LEFT_ANKLE: (0.2, 2.6),
    Joint.RIGHT_ANKLE: (-0.2, 2.6),
}


def standing(origin=(320.0, 150.0), torso=100.0, **overrides) -> np.ndarray:
    """(NUM_JOINTS, 2) pixel positions; overrides map joint names to body offsets."""
    offsets = dict(STANDING)
    for name, offset in overrides.items():
        offsets[Joint[name.upper()]] = offset
    return np.array([offsets[j] for j in Joint]) * torso + np.asarray(origin)


def raw(points: np.ndarray, confidence: float = 0.95) -> RawPose:
    return RawPose(points, np.full(NUM_JOINTS, confidence))
