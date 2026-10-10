"""Temporal filtering of joint positions.

Every joint runs through a small state machine so the game never receives raw jitter, single-frame
glitches, or a sudden jump when a joint is occluded:

    LOST      --confident detection-------------------> ACQUIRING
    ACQUIRING --`acquire_frames` consistent detections--> TRACKED
    TRACKED   --low confidence or implausible jump------> PREDICTED
    PREDICTED --consistent detection--------------------> TRACKED
    PREDICTED --no detection for `max_gap` seconds------> LOST

Confidence uses hysteresis (`confidence_on` > `confidence_off`) so a joint hovering around one
threshold does not flicker between states. While PREDICTED the joint keeps moving with its last
velocity, decaying exponentially, so short occlusions are bridged without the joint either freezing
or drifting away.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

MIN_DT = 1e-3


class JointState(IntEnum):
    LOST = 0
    ACQUIRING = 1
    PREDICTED = 2
    TRACKED = 3


def smoothing_factor(cutoff: float | np.ndarray, dt: float) -> float | np.ndarray:
    """Weight of a new sample in a first-order low-pass with `cutoff` Hz sampled every `dt` s."""
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """One Euro filter (Casiez, Roussel & Vogel, CHI 2012) over an (N, D) array of points.

    A low-pass filter whose cutoff rises with speed: at rest it stays at `min_cutoff` and removes
    jitter, during fast motion it opens up by `beta` Hz per unit of speed so lag stays small. Speed
    is the norm of each point's velocity divided by `scale`, so with scale = torso length `beta` is
    per body length per second and the filter behaves the same at any distance from the camera.
    """

    def __init__(self, shape: tuple[int, int], min_cutoff: float, beta: float, d_cutoff: float):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.value = np.zeros(shape)
        self.derivative = np.zeros(shape)

    def set(self, rows: np.ndarray, value: np.ndarray, derivative: np.ndarray | float = 0.0) -> None:
        self.value[rows] = value
        self.derivative[rows] = derivative

    def update(self, x: np.ndarray, dt: float, rows: np.ndarray, scale: float = 1.0) -> None:
        """Filter the samples `x[rows]` into the selected rows; other rows are left untouched."""
        sample, previous = x[rows], self.value[rows]
        derivative = self.derivative[rows]
        derivative += smoothing_factor(self.d_cutoff, dt) * ((sample - previous) / dt - derivative)
        speed = np.linalg.norm(derivative, axis=1, keepdims=True) / scale
        cutoff = self.min_cutoff + self.beta * speed
        self.value[rows] = previous + smoothing_factor(cutoff, dt) * (sample - previous)
        self.derivative[rows] = derivative


@dataclass(frozen=True)
class FilterConfig:
    min_cutoff: float = 1.0  # Hz at rest; lower removes more jitter
    beta: float = 1.0  # Hz added per body length/s of speed; higher reduces lag during motion
    d_cutoff: float = 1.0  # Hz, smoothing of the speed estimate that drives the cutoff
    velocity_cutoff: float = 4.0  # Hz, smoothing of the reported velocity
    confidence_on: float = 0.6  # a lost joint needs this confidence to count as detected
    confidence_off: float = 0.4  # a tracked joint is dropped below this confidence
    acquire_frames: int = 3  # consecutive consistent detections before a lost joint is trusted
    max_speed: float = 15.0  # body lengths/s; detections implying faster motion are rejected
    gate_margin: float = 0.3  # body lengths always tolerated on top of max_speed * elapsed time
    max_gap: float = 0.4  # s a joint is predicted through before it is declared lost
    prediction_decay: float = 0.1  # s, velocity time constant while predicting; 0 holds position


class KeypointFilter:
    """Turns noisy per-frame detections of N joints into continuous, smooth estimates."""

    def __init__(self, num_joints: int, config: FilterConfig = FilterConfig()):
        self.config = config
        self.velocity = np.zeros((num_joints, 2))
        self.state = np.full(num_joints, JointState.LOST, dtype=np.int8)
        self.measured = np.zeros(num_joints, dtype=bool)
        self._euro = OneEuroFilter((num_joints, 2), config.min_cutoff, config.beta, config.d_cutoff)
        self._streak = np.zeros(num_joints, dtype=int)
        self._last_seen = np.full(num_joints, -np.inf)
        self._time: float | None = None

    @property
    def position(self) -> np.ndarray:
        return self._euro.value

    def update(self, points: np.ndarray | None, confidence: np.ndarray | None, t: float, scale: float) -> None:
        """Advance to time `t` (s). `points` is None when the player was not detected this frame;
        `scale` is the body length in pixels used to express speeds and gates in body lengths."""
        cfg = self.config
        dt = MIN_DT if self._time is None else max(t - self._time, MIN_DT)
        self._time = t

        state = self.state
        live = (state == JointState.TRACKED) | (state == JointState.PREDICTED)
        acquiring = state == JointState.ACQUIRING
        lost = state == JointState.LOST

        if points is None:
            confident = np.zeros_like(live)
            points = self.position
        else:
            confident = confidence >= np.where(live, cfg.confidence_off, cfg.confidence_on)

        allowed = (cfg.max_speed * (t - self._last_seen) + cfg.gate_margin) * scale
        consistent = np.linalg.norm(points - self.position, axis=1) <= allowed

        accepted = confident & consistent & ~lost
        restart = confident & (lost | (acquiring & ~consistent))
        coast = live & ~accepted
        dropped = acquiring & ~confident

        if accepted.any():
            previous = self.position[accepted]
            self._euro.update(points, dt, accepted, scale)
            measured_velocity = (self.position[accepted] - previous) / dt
            velocity = self.velocity[accepted]
            self.velocity[accepted] = velocity + smoothing_factor(cfg.velocity_cutoff, dt) * (measured_velocity - velocity)

        if coast.any():
            decay = np.exp(-dt / cfg.prediction_decay) if cfg.prediction_decay > 0 else 0.0
            self.velocity[coast] *= decay
            self._euro.set(coast, self.position[coast] + self.velocity[coast] * dt, self.velocity[coast])

        if restart.any():
            self._euro.set(restart, points[restart])
            self.velocity[restart] = 0.0

        self._streak[accepted] += 1
        self._streak[restart] = 1
        self._last_seen[accepted | restart] = t

        state[accepted & live] = JointState.TRACKED
        state[restart] = JointState.ACQUIRING
        state[(state == JointState.ACQUIRING) & (self._streak >= cfg.acquire_frames)] = JointState.TRACKED
        state[coast] = JointState.PREDICTED

        gone = dropped | (coast & (t - self._last_seen > cfg.max_gap))
        state[gone] = JointState.LOST
        self.velocity[gone] = 0.0
        self._streak[gone] = 0

        self.measured = accepted | restart
