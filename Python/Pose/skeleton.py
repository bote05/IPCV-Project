"""Joints used by the climbing game and their place in MediaPipe's 33-landmark BlazePose topology.

All positions in this package are in pixels of the processed frame (x right, y down), so distances
are isotropic. Derived motion signals are expressed in body lengths (see motion.py) and are
therefore independent of resolution and of the player's distance to the camera.
"""

from enum import IntEnum


class Joint(IntEnum):
    NOSE = 0
    LEFT_SHOULDER = 1
    RIGHT_SHOULDER = 2
    LEFT_ELBOW = 3
    RIGHT_ELBOW = 4
    LEFT_WRIST = 5
    RIGHT_WRIST = 6
    LEFT_HIP = 7
    RIGHT_HIP = 8
    LEFT_KNEE = 9
    RIGHT_KNEE = 10
    LEFT_ANKLE = 11
    RIGHT_ANKLE = 12


NUM_JOINTS = len(Joint)
JOINT_NAMES = tuple(joint.name.lower() for joint in Joint)

MEDIAPIPE_INDEX = (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)

SIDES = ("left", "right")
SHOULDERS = (Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER)
ELBOWS = (Joint.LEFT_ELBOW, Joint.RIGHT_ELBOW)
WRISTS = (Joint.LEFT_WRIST, Joint.RIGHT_WRIST)
HIPS = (Joint.LEFT_HIP, Joint.RIGHT_HIP)
KNEES = (Joint.LEFT_KNEE, Joint.RIGHT_KNEE)
ANKLES = (Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE)

BONES = (
    (Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER),
    (Joint.LEFT_HIP, Joint.RIGHT_HIP),
    *zip(SHOULDERS, ELBOWS),
    *zip(ELBOWS, WRISTS),
    *zip(SHOULDERS, HIPS),
    *zip(HIPS, KNEES),
    *zip(KNEES, ANKLES),
)
