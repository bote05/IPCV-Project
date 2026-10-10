"""Convert the team's tracking output to the bridge format."""

from dataclasses import dataclass, field
from .protocol import Joint, JointState, Landmark, MotionEvent, PlayerState


@dataclass
class TrackingResult:
    players: tuple[PlayerState, ...] = ()
    face_jpegs: dict[int, bytes] = field(default_factory=dict)


def player_from_pose(data: dict) -> PlayerState:
    """Adapt PlayerPose.to_dict(); normalization and display mirroring happen in Pose."""
    if type(data["id"]) is not int or data["id"] not in (1, 2):
        raise ValueError("Assign player IDs 1 or 2 before updating PoseTracker")
    if [joint["name"] for joint in data["joints"]] != [joint.name.lower() for joint in Joint]:
        raise ValueError("Pose must contain the 13 joints in the agreed order")
    # Missing this frame is not yet lost: Pose can predict through short gaps.
    if not data["valid"]:
        return PlayerState(data["id"], False)
    pose = tuple(
        Landmark((joint["x"], joint["y"]), JointState(joint["state"]))
        for joint in data["joints"]
    )
    return PlayerState(data["id"], True, pose=pose, hand_raised=tuple(data["hand_raised"]),
                       events=tuple(MotionEvent(**event) for event in data["events"]))
