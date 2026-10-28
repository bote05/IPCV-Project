import numpy as np
import pytest

from Pose.detector import RawPose
from Pose.skeleton import NUM_JOINTS, Joint

from .poses import standing


def test_raw_pose_holds_read_only_copies():
    points, confidence = standing(), np.full(NUM_JOINTS, 0.9)
    pose = RawPose(points, confidence)
    points[:] = 0.0
    confidence[:] = 0.0
    np.testing.assert_array_equal(pose.points, standing())
    assert (pose.confidence == 0.9).all()
    for array in (pose.points, pose.confidence):
        with pytest.raises(ValueError):
            array[0] = 1.0


def test_hiding_joints_leaves_the_original_pose_unchanged():
    pose = RawPose(standing(), np.full(NUM_JOINTS, 0.9))
    hidden = pose.hide([Joint.LEFT_WRIST])
    assert hidden.confidence[Joint.LEFT_WRIST] == 0.0
    assert pose.confidence[Joint.LEFT_WRIST] == 0.9
    np.testing.assert_array_equal(hidden.points, pose.points)
    with pytest.raises(ValueError):
        hidden.points[0] = 0.0
