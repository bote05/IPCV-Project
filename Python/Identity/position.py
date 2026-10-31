
class PositionEstimator:
    def __init__(self, fx, fy, cx, cy):
        # Camera parameters.
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)

        # Calibration constant for each player.
        self.K = {}

        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("Focal lengths must be positive.")

    def estimate(self, player_pose, known_distance=None):

        player_id = player_pose.id
        motion = player_pose.motion

        if not motion.valid or motion.scale <= 0:
            return None

        # Calibrate the player if not already calibrated.
        if player_id not in self.K:
            if known_distance is None or known_distance <= 0:
                return None

            self.K[player_id] = float(known_distance * motion.scale)

        # Estimate depth.
        z = self.K[player_id] / motion.scale

        # Hip centre in pixel coordinates.
        u, v = motion.center

        # Convert to camera-relative coordinates in metres.
        x = (u - self.cx) * z / self.fx
        y = (self.cy - v) * z / self.fy

        return float(x), float(y), float(z)

    def estimate_players(self, player_poses, known_distance=None):

        positions = {}

        for player_pose in player_poses:
            if player_pose.id is None:
                continue

            position = self.estimate(
                player_pose,
                known_distance
            )

            if position is not None:
                positions[player_pose.id] = position

        return positions

    def test_distance(self, player_poses, known_distance):
        results = {}
        for player_pose in player_poses:
            if player_pose.id is None:
                continue

            estimated_position = self.estimate(player_pose)

            if estimated_position is None:
                continue

            estimated_distance = estimated_position[2]
            absolute_error = abs(estimated_distance-known_distance)
            percentage_error = (absolute_error/known_distance)
            results[player_pose.id] = {
            "actual_distance": known_distance,
            "estimated_distance": estimated_distance,
            "absolute_error": absolute_error,
            "percentage_error": percentage_error
            }
        return results
