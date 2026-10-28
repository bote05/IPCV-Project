"""Task 2: body pose estimation and motion tracking.

    detector = PoseDetector("full", num_poses=2)
    tracker = PoseTracker()
    for frame, t in camera:
        players = tracker.update(detector.detect(frame, t), t)
        messages = [player.to_dict((width, height)) for player in players]
"""

from .detector import PoseDetector, RawPose
from .filtering import FilterConfig, JointState, KeypointFilter, OneEuroFilter
from .motion import MotionAnalyzer, MotionConfig, MotionEvent, MotionSignals
from .skeleton import JOINT_NAMES, Joint
from .tracker import PlayerPose, PoseTracker, TrackerConfig

__all__ = [
    "FilterConfig",
    "JOINT_NAMES",
    "Joint",
    "JointState",
    "KeypointFilter",
    "MotionAnalyzer",
    "MotionConfig",
    "MotionEvent",
    "MotionSignals",
    "OneEuroFilter",
    "PlayerPose",
    "PoseDetector",
    "PoseTracker",
    "RawPose",
    "TrackerConfig",
]
