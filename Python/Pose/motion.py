"""Control signals derived from filtered joints.

Signals are expressed in a body frame: origin at the shoulder midpoint, x to the right of the image,
y up, unit = torso length (shoulder midpoint to hip midpoint). They therefore do not change when a
player stands elsewhere in the image or closer to the camera, and one set of thresholds works for
both players.

Continuous signals are computed from TRACKED and PREDICTED joints, so they stay continuous through
short occlusions. Discrete outputs (hand raised, moving, pull and jump events) only change on
TRACKED joints, so an uncertain or extrapolated joint can never trigger a new game action; when a
joint is lost they fall back to their neutral value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .filtering import JointState
from .skeleton import ELBOWS, HIPS, SHOULDERS, SIDES, WRISTS

TORSO_PER_SHOULDER_WIDTH = 1.3  # adult proportions seen from the front
_Y_UP = np.array([1.0, -1.0])


@dataclass(frozen=True)
class MotionConfig:
    raise_on: float = 0.6  # wrist height above the shoulders (torso lengths) to count as raised
    raise_off: float = 0.4
    moving_on: float = 0.8  # joint speed (torso lengths/s) to count as moving
    moving_off: float = 0.4
    pull_distance: float = 0.5  # downward wrist travel from its last high point that fires a pull
    pull_speed: float = 1.0  # minimum downward wrist speed (torso lengths/s) for that travel
    pull_min_height: float = 0.0  # the high point must be above this height (0 = shoulder level)
    jump_speed: float = 2.0  # upward hip speed (torso lengths/s) that fires a jump
    jump_max_stretch: float = 0.3  # 1/s; faster torso-length change means walking in depth, not jumping
    scale_time_constant: float = 0.5  # s, smoothing of the torso length estimate


@dataclass(frozen=True)
class MotionEvent:
    kind: str  # "pull" or "jump"
    side: str | None  # "left" / "right" for pulls
    time: float
    strength: float  # pull: travel in torso lengths; jump: hip speed in torso lengths/s


@dataclass(frozen=True)
class MotionSignals:
    valid: bool  # body frame available: both shoulders tracked or predicted
    scale: float  # torso length in pixels
    center: np.ndarray  # (2,) hip midpoint in pixels
    center_velocity: np.ndarray  # (2,) torso lengths/s, y up
    lean: float  # shoulder-line roll in degrees, counter-clockwise on screen
    body: np.ndarray  # (N, 2) joints in the body frame
    body_velocity: np.ndarray  # (N, 2) torso lengths/s relative to the shoulders, y up
    moving: np.ndarray  # (N,) bool
    arm_extension: np.ndarray  # (2,) left, right: 1 = straight arm, ~0.5 = elbow fully bent
    hand_raised: np.ndarray  # (2,) left, right
    events: tuple[MotionEvent, ...]


def torso_length(points: np.ndarray, usable: np.ndarray) -> float | None:
    """Shoulder-midpoint to hip-midpoint distance, falling back to one side of the torso or to the
    shoulder width when the hips are not visible."""
    shoulders, hips = list(SHOULDERS), list(HIPS)
    if usable[shoulders + hips].all():
        return float(np.linalg.norm(points[shoulders].mean(axis=0) - points[hips].mean(axis=0)))
    sides = [np.linalg.norm(points[s] - points[h]) for s, h in zip(SHOULDERS, HIPS) if usable[s] and usable[h]]
    if sides:
        return float(np.mean(sides))
    if usable[shoulders].all():
        return float(np.linalg.norm(points[SHOULDERS[0]] - points[SHOULDERS[1]])) * TORSO_PER_SHOULDER_WIDTH
    return None


class MotionAnalyzer:
    """Converts one player's filtered joints into body-relative control signals and motion events."""

    def __init__(self, num_joints: int, config: MotionConfig = MotionConfig()):
        self.config = config
        self.scale: float | None = None
        self._time: float | None = None
        self._body = np.zeros((num_joints, 2))
        self._body_velocity = np.zeros((num_joints, 2))
        self._moving = np.zeros(num_joints, dtype=bool)
        self._center = np.zeros(2)
        self._center_velocity = np.zeros(2)
        self._lean = 0.0
        self._arm_extension = np.ones(2)
        self._hand_raised = np.zeros(2, dtype=bool)
        self._pull_anchor = np.full(2, np.nan)
        self._pull_fired = np.zeros(2, dtype=bool)
        self._jump_armed = True

    def update(self, position: np.ndarray, velocity: np.ndarray, state: np.ndarray, t: float) -> MotionSignals:
        dt = 0.0 if self._time is None else t - self._time
        self._time = t

        usable = (state == JointState.TRACKED) | (state == JointState.PREDICTED)
        confirmed = state == JointState.TRACKED
        self._update_scale(position, usable, dt)
        valid = self.scale is not None and bool(usable[list(SHOULDERS)].all())
        events: list[MotionEvent] = []

        if self.scale is not None:
            speed = np.linalg.norm(velocity, axis=1) / self.scale
            threshold = np.where(self._moving, self.config.moving_off, self.config.moving_on)
            self._moving = np.where(confirmed, speed > threshold, self._moving & usable)
            self._update_center(position, velocity, usable)
            if confirmed[list(SHOULDERS) + list(HIPS)].all():
                events += self._detect_jump(position, velocity, t)
            elif self._center_velocity[1] <= 0:
                self._jump_armed = True

        if valid:
            self._update_body_frame(position, velocity, usable)
            self._update_arms(usable, confirmed)
            events += self._detect_pulls(confirmed, t)
        else:
            self._body_velocity[:] = 0.0
            self._hand_raised[:] = False
            self._pull_anchor[:] = np.nan

        return MotionSignals(
            valid=valid,
            scale=self.scale or 0.0,
            center=self._center.copy(),
            center_velocity=self._center_velocity.copy(),
            lean=self._lean,
            body=self._body.copy(),
            body_velocity=self._body_velocity.copy(),
            moving=self._moving.copy(),
            arm_extension=self._arm_extension.copy(),
            hand_raised=self._hand_raised.copy(),
            events=tuple(events),
        )

    def _update_scale(self, position: np.ndarray, usable: np.ndarray, dt: float) -> None:
        measured = torso_length(position, usable)
        if measured is None:
            return
        if self.scale is None:
            self.scale = measured
        else:
            self.scale += (1.0 - np.exp(-dt / self.config.scale_time_constant)) * (measured - self.scale)

    def _update_center(self, position: np.ndarray, velocity: np.ndarray, usable: np.ndarray) -> None:
        hips, shoulders = list(HIPS), list(SHOULDERS)
        if usable[hips].all():
            self._center = position[hips].mean(axis=0)
            self._center_velocity = velocity[hips].mean(axis=0) * _Y_UP / self.scale
        elif usable[shoulders].all():
            self._center = position[shoulders].mean(axis=0) + (0.0, self.scale)
            self._center_velocity = velocity[shoulders].mean(axis=0) * _Y_UP / self.scale
        else:
            self._center_velocity = np.zeros(2)

    def _update_body_frame(self, position: np.ndarray, velocity: np.ndarray, usable: np.ndarray) -> None:
        shoulders = list(SHOULDERS)
        origin = position[shoulders].mean(axis=0)
        origin_velocity = velocity[shoulders].mean(axis=0)
        self._body[usable] = (position[usable] - origin) * _Y_UP / self.scale
        self._body_velocity[usable] = (velocity[usable] - origin_velocity) * _Y_UP / self.scale
        self._body_velocity[~usable] = 0.0

        across = self._body[SHOULDERS[0]] - self._body[SHOULDERS[1]]
        # Measure from the screen-left to the screen-right shoulder so mirroring does not flip the angle.
        self._lean = float(np.degrees(np.arctan2(across[1] * np.sign(across[0]), abs(across[0]))))

    def _update_arms(self, usable: np.ndarray, confirmed: np.ndarray) -> None:
        cfg = self.config
        for i, (shoulder, elbow, wrist) in enumerate(zip(SHOULDERS, ELBOWS, WRISTS)):
            if usable[[shoulder, elbow, wrist]].all():
                upper = np.linalg.norm(self._body[elbow] - self._body[shoulder])
                lower = np.linalg.norm(self._body[wrist] - self._body[elbow])
                reach = np.linalg.norm(self._body[wrist] - self._body[shoulder])
                self._arm_extension[i] = reach / max(upper + lower, 1e-6)

            if confirmed[wrist]:
                threshold = cfg.raise_off if self._hand_raised[i] else cfg.raise_on
                self._hand_raised[i] = self._body[wrist, 1] > threshold
            elif not usable[wrist]:
                self._hand_raised[i] = False

    def _detect_pulls(self, confirmed: np.ndarray, t: float) -> list[MotionEvent]:
        """A pull is a fast downward wrist stroke relative to the shoulders, the climbing motion of
        pulling the body up a hold. The anchor follows the wrist while it rises or rests and freezes
        when it starts descending; the travel below the anchor is the stroke length."""
        cfg = self.config
        events = []
        for i, (side, wrist) in enumerate(zip(SIDES, WRISTS)):
            if not confirmed[wrist]:
                self._pull_anchor[i] = np.nan
                continue
            height, vertical_speed = self._body[wrist, 1], self._body_velocity[wrist, 1]
            if vertical_speed >= 0:
                self._pull_anchor[i] = height
                self._pull_fired[i] = False
                continue
            travel = self._pull_anchor[i] - height
            if (
                not self._pull_fired[i]
                and travel >= cfg.pull_distance
                and -vertical_speed >= cfg.pull_speed
                and self._pull_anchor[i] >= cfg.pull_min_height
            ):
                events.append(MotionEvent("pull", side, t, float(travel)))
                self._pull_fired[i] = True
        return events

    def _detect_jump(self, position: np.ndarray, velocity: np.ndarray, t: float) -> list[MotionEvent]:
        """A jump is a fast upward hip movement at constant torso length. Walking away from a
        camera mounted above hip height also moves the hips up in the image, but shrinks the torso.
        The stretch rate comes from the same filtered velocities as the hip speed, so both react
        equally fast."""
        cfg = self.config
        shoulders, hips = list(SHOULDERS), list(HIPS)
        torso = position[shoulders].mean(axis=0) - position[hips].mean(axis=0)
        torso_velocity = velocity[shoulders].mean(axis=0) - velocity[hips].mean(axis=0)
        stretch = torso @ torso_velocity / max(torso @ torso, 1e-6)
        upward_speed = self._center_velocity[1]
        if self._jump_armed and abs(stretch) <= cfg.jump_max_stretch and upward_speed >= cfg.jump_speed:
            self._jump_armed = False
            return [MotionEvent("jump", None, t, float(upward_speed))]
        if upward_speed <= 0:
            self._jump_armed = True
        return []
