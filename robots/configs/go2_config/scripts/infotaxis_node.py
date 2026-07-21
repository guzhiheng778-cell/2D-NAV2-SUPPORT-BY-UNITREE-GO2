#!/usr/bin/env python3

import math
from typing import Dict, List, Optional, Tuple

import rclpy
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped, Twist, Vector3Stamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import ColorRGBA, Float32MultiArray, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def quaternion_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw: float):
    half_yaw = 0.5 * yaw
    return 0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)


class Infotaxis(Node):
    def __init__(self):
        super().__init__("infotaxis")

        self.declare_parameter("robot_pose_topic", "/amcl_pose")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("allow_odom_pose_fallback", False)
        self.declare_parameter("probability_grid_topic", "/probability_grid")
        self.declare_parameter("probability_array_topic", "/probability_values")
        self.declare_parameter("wind_topic", "/wind_vector")
        self.declare_parameter("costmap_topic", "/local_costmap/costmap")
        self.declare_parameter("velocity_topic", "/v_info")
        self.declare_parameter("best_goal_topic", "/best_info_goal")
        self.declare_parameter("status_topic", "/infotaxis/status")
        self.declare_parameter("marker_topic", "/infotaxis/markers")
        self.declare_parameter("control_rate_hz", 2.0)

        self.declare_parameter("candidate_distance", 0.90)
        self.declare_parameter("candidate_distances", [0.90, 0.65, 0.40])
        self.declare_parameter(
            "candidate_yaw_offsets",
            [0.0, math.pi / 4.0, -math.pi / 4.0, math.pi / 2.0, -math.pi / 2.0, math.pi],
        )
        self.declare_parameter("include_stay_candidate", True)
        self.declare_parameter("candidate_radius", 0.08)

        self.declare_parameter("max_linear_vel", 0.18)
        self.declare_parameter("max_angular_vel", 0.45)
        self.declare_parameter("angular_gain", 0.90)
        self.declare_parameter("rotate_in_place_angle", 1.20)
        self.declare_parameter("velocity_smoothing_alpha", 0.25)
        self.declare_parameter("publish_zero_without_inputs", True)

        self.declare_parameter("release_rate", 8.0)
        self.declare_parameter("plume_sigma_y_base", 0.20)
        self.declare_parameter("plume_sigma_y_gain", 0.22)
        self.declare_parameter("plume_sigma_z_base", 0.20)
        self.declare_parameter("plume_sigma_z_gain", 0.12)
        self.declare_parameter("sensor_height", 0.25)
        self.declare_parameter("source_height", 0.2)
        self.declare_parameter("detection_threshold", 0.18)
        self.declare_parameter("likelihood_temperature", 0.12)

        self.declare_parameter("obstacle_lethal_threshold", 85)
        self.declare_parameter("obstacle_unknown_cost", 0.35)
        self.declare_parameter("reject_unknown_goals", False)
        self.declare_parameter("invalid_candidate_delta", -1.0e9)

        self.candidate_distance = float(self.get_parameter("candidate_distance").value)
        self.allow_odom_pose_fallback = bool(
            self.get_parameter("allow_odom_pose_fallback").value
        )
        configured_distances = [
            float(value) for value in self.get_parameter("candidate_distances").value
        ]
        self.candidate_distances = [
            distance for distance in configured_distances if distance > 0.0
        ] or [self.candidate_distance]
        self.candidate_yaw_offsets = [
            float(value) for value in self.get_parameter("candidate_yaw_offsets").value
        ]
        self.include_stay_candidate = bool(
            self.get_parameter("include_stay_candidate").value
        )
        self.candidate_radius = float(self.get_parameter("candidate_radius").value)
        self.max_linear_vel = float(self.get_parameter("max_linear_vel").value)
        self.max_angular_vel = float(self.get_parameter("max_angular_vel").value)
        self.angular_gain = float(self.get_parameter("angular_gain").value)
        self.rotate_in_place_angle = float(self.get_parameter("rotate_in_place_angle").value)
        self.velocity_smoothing_alpha = clamp(
            float(self.get_parameter("velocity_smoothing_alpha").value), 0.0, 1.0
        )
        self.publish_zero_without_inputs = bool(
            self.get_parameter("publish_zero_without_inputs").value
        )
        self.release_rate = float(self.get_parameter("release_rate").value)
        self.sigma_y_base = float(self.get_parameter("plume_sigma_y_base").value)
        self.sigma_y_gain = float(self.get_parameter("plume_sigma_y_gain").value)
        self.sigma_z_base = float(self.get_parameter("plume_sigma_z_base").value)
        self.sigma_z_gain = float(self.get_parameter("plume_sigma_z_gain").value)
        self.sensor_height = float(self.get_parameter("sensor_height").value)
        self.source_height = float(self.get_parameter("source_height").value)
        self.detection_threshold = float(self.get_parameter("detection_threshold").value)
        self.likelihood_temperature = max(
            1e-3, float(self.get_parameter("likelihood_temperature").value)
        )
        self.obstacle_lethal_threshold = int(
            self.get_parameter("obstacle_lethal_threshold").value
        )
        self.obstacle_unknown_cost = float(self.get_parameter("obstacle_unknown_cost").value)
        self.reject_unknown_goals = bool(self.get_parameter("reject_unknown_goals").value)
        self.invalid_candidate_delta = float(
            self.get_parameter("invalid_candidate_delta").value
        )

        self.robot_pose: Optional[Tuple[float, float, float]] = None
        self.has_map_pose = False
        self.probability_grid: Optional[OccupancyGrid] = None
        self.probability_values: Optional[List[float]] = None
        self.wind_xy = (1.0, 0.0)
        self.costmap: Optional[OccupancyGrid] = None
        self.last_scores: List[Dict[str, float]] = []
        self.filtered_cmd = Twist()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.transform_cache = {}

        self.velocity_pub = self.create_publisher(
            Twist, self.get_parameter("velocity_topic").value, 10
        )
        self.best_goal_pub = self.create_publisher(
            PoseStamped, self.get_parameter("best_goal_topic").value, 10
        )
        self.status_pub = self.create_publisher(
            String, self.get_parameter("status_topic").value, 10
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, self.get_parameter("marker_topic").value, 10
        )

        self.create_subscription(
            PoseWithCovarianceStamped,
            self.get_parameter("robot_pose_topic").value,
            self._pose_callback,
            10,
        )
        self.create_subscription(
            Odometry, self.get_parameter("odom_topic").value, self._odom_callback, 20
        )
        self.create_subscription(
            OccupancyGrid,
            self.get_parameter("probability_grid_topic").value,
            self._probability_grid_callback,
            2,
        )
        self.create_subscription(
            Float32MultiArray,
            self.get_parameter("probability_array_topic").value,
            self._probability_array_callback,
            2,
        )
        self.create_subscription(
            Vector3Stamped, self.get_parameter("wind_topic").value, self._wind_callback, 10
        )
        self.create_subscription(
            OccupancyGrid, self.get_parameter("costmap_topic").value, self._costmap_callback, 2
        )

        rate = max(0.5, float(self.get_parameter("control_rate_hz").value))
        self.timer = self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            "Strict infotaxis ready: selecting candidate with maximum expected entropy drop"
        )

    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        self.has_map_pose = True
        self.robot_pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            quaternion_to_yaw(msg.pose.pose.orientation),
        )

    def _odom_callback(self, msg: Odometry):
        if self.has_map_pose or not self.allow_odom_pose_fallback:
            return
        self.robot_pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            quaternion_to_yaw(msg.pose.pose.orientation),
        )

    def _probability_grid_callback(self, msg: OccupancyGrid):
        self.probability_grid = msg

    def _probability_array_callback(self, msg: Float32MultiArray):
        self.probability_values = [float(value) for value in msg.data]

    def _wind_callback(self, msg: Vector3Stamped):
        self.wind_xy = (msg.vector.x, msg.vector.y)

    def _costmap_callback(self, msg: OccupancyGrid):
        self.costmap = msg

    def _tick(self):
        self.transform_cache = {}
        posterior = self._posterior()
        if self.robot_pose is None or self.probability_grid is None or posterior is None:
            self._publish_status("waiting_for_pose_or_probability_distribution")
            if self.publish_zero_without_inputs:
                self.velocity_pub.publish(Twist())
            return

        entropy_now = self._entropy(posterior)
        candidates = self._candidate_points()
        scored = [
            self._score_candidate(candidate, posterior, entropy_now)
            for candidate in candidates
        ]
        valid = [item for item in scored if item["valid"]]
        self.last_scores = scored

        if not valid:
            self.filtered_cmd = Twist()
            self.velocity_pub.publish(Twist())
            self.marker_pub.publish(self._markers(scored, None))
            reason_counts = {}
            for item in scored:
                reason = item.get("invalid_reason", "unknown")
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            reason_text = " ".join(
                f"{reason}={count}" for reason, count in sorted(reason_counts.items())
            )
            self._publish_status(
                f"no_valid_infotaxis_candidate total={len(scored)} {reason_text}".rstrip()
            )
            return

        best = max(valid, key=lambda item: item["delta_entropy"])
        cmd = self._smooth_velocity(self._velocity_for_candidate(best))
        self.velocity_pub.publish(cmd)
        self.best_goal_pub.publish(self._best_goal_msg(best))
        self.marker_pub.publish(self._markers(scored, best))
        self._publish_status(
            "best={name} r={distance:.2f} dS={delta_entropy:.6f} S_now={entropy_now:.6f} "
            "E[S]={expected_entropy:.6f} P_hit={p_hit:.3f} "
            "S_hit={entropy_hit:.6f} S_void={entropy_void:.6f}".format(**best)
        )

    def _posterior(self) -> Optional[List[float]]:
        if self.probability_grid is None:
            return None

        cell_count = int(self.probability_grid.info.width * self.probability_grid.info.height)
        if self.probability_values is not None and len(self.probability_values) == cell_count:
            values = [max(0.0, value) for value in self.probability_values]
        else:
            values = [max(0.0, float(value)) for value in self.probability_grid.data]

        total = sum(values)
        if total <= 0.0 or not math.isfinite(total):
            return [1.0 / cell_count] * cell_count
        return [value / total for value in values]

    def _candidate_points(self) -> List[Dict[str, float]]:
        robot_x, robot_y, robot_yaw = self.robot_pose
        candidates = []
        for distance in self.candidate_distances:
            for offset in self.candidate_yaw_offsets:
                heading = normalize_angle(robot_yaw + offset)
                candidates.append(
                    {
                        "name": self._candidate_name(offset),
                        "x": robot_x + distance * math.cos(heading),
                        "y": robot_y + distance * math.sin(heading),
                        "heading": heading,
                        "relative_yaw": offset,
                        "distance": distance,
                    }
                )

        if self.include_stay_candidate:
            candidates.append(
                {
                    "name": "stay",
                    "x": robot_x,
                    "y": robot_y,
                    "heading": robot_yaw,
                    "relative_yaw": 0.0,
                    "distance": 0.0,
                }
            )
        return candidates

    def _candidate_name(self, offset: float) -> str:
        if abs(offset) < 1e-3:
            return "front"
        if abs(offset - math.pi / 4.0) < 1e-3:
            return "left_front"
        if abs(offset + math.pi / 4.0) < 1e-3:
            return "right_front"
        if abs(offset - math.pi / 2.0) < 1e-3:
            return "left"
        if abs(offset + math.pi / 2.0) < 1e-3:
            return "right"
        if abs(abs(offset) - math.pi) < 1e-3:
            return "back"
        return f"yaw_{offset:+.2f}"

    def _score_candidate(
        self,
        candidate: Dict[str, float],
        posterior: List[float],
        entropy_now: float,
    ) -> Dict[str, float]:
        result = dict(candidate)
        result["entropy_now"] = entropy_now

        traversable, invalid_reason = self._goal_traversability(
            candidate["x"], candidate["y"]
        )
        if not traversable:
            result.update(
                {
                    "valid": False,
                    "invalid_reason": invalid_reason,
                    "p_hit": 0.0,
                    "p_void": 0.0,
                    "entropy_hit": entropy_now,
                    "entropy_void": entropy_now,
                    "expected_entropy": entropy_now,
                    "delta_entropy": self.invalid_candidate_delta,
                }
            )
            return result

        hit_likelihoods = []
        void_likelihoods = []
        p_hit = 0.0
        p_void = 0.0

        for index, prior in enumerate(posterior):
            source_x, source_y = self._cell_center(index)
            hit_probability = self._hit_probability(
                source_x, source_y, candidate["x"], candidate["y"]
            )
            void_probability = 1.0 - hit_probability
            hit_likelihoods.append(hit_probability)
            void_likelihoods.append(void_probability)
            p_hit += prior * hit_probability
            p_void += prior * void_probability

        entropy_hit = self._posterior_entropy_after_observation(
            posterior, hit_likelihoods, p_hit, entropy_now
        )
        entropy_void = self._posterior_entropy_after_observation(
            posterior, void_likelihoods, p_void, entropy_now
        )
        expected_entropy = p_hit * entropy_hit + p_void * entropy_void
        delta_entropy = entropy_now - expected_entropy

        result.update(
            {
                "valid": True,
                "invalid_reason": "none",
                "p_hit": p_hit,
                "p_void": p_void,
                "entropy_hit": entropy_hit,
                "entropy_void": entropy_void,
                "expected_entropy": expected_entropy,
                "delta_entropy": delta_entropy,
            }
        )
        return result

    def _hit_probability(
        self,
        source_x: float,
        source_y: float,
        observer_x: float,
        observer_y: float,
    ) -> float:
        concentration = self._expected_concentration(
            source_x, source_y, observer_x, observer_y
        )
        exponent = -(concentration - self.detection_threshold) / self.likelihood_temperature
        return 1.0 / (1.0 + math.exp(clamp(exponent, -60.0, 60.0)))

    def _expected_concentration(
        self,
        source_x: float,
        source_y: float,
        observer_x: float,
        observer_y: float,
    ) -> float:
        wind_x, wind_y = self.wind_xy
        wind_speed = max(0.05, math.hypot(wind_x, wind_y))
        wind_yaw = math.atan2(wind_y, wind_x)

        dx = observer_x - source_x
        dy = observer_y - source_y
        downwind = dx * math.cos(wind_yaw) + dy * math.sin(wind_yaw)
        crosswind = -dx * math.sin(wind_yaw) + dy * math.cos(wind_yaw)

        if downwind <= 0.0:
            return 0.0

        sigma_y = self.sigma_y_base + self.sigma_y_gain * downwind
        sigma_z = self.sigma_z_base + self.sigma_z_gain * downwind
        vertical = self.sensor_height - self.source_height
        plume = self.release_rate / (2.0 * math.pi * wind_speed * sigma_y * sigma_z)
        plume *= math.exp(-0.5 * (crosswind / sigma_y) ** 2)
        plume *= math.exp(-0.5 * (vertical / sigma_z) ** 2)
        return max(0.0, plume)

    def _posterior_entropy_after_observation(
        self,
        posterior: List[float],
        likelihoods: List[float],
        evidence: float,
        fallback_entropy: float,
    ) -> float:
        if evidence <= 1e-15 or not math.isfinite(evidence):
            return fallback_entropy

        entropy = 0.0
        for prior, likelihood in zip(posterior, likelihoods):
            probability = prior * likelihood / evidence
            if probability > 0.0:
                entropy -= probability * math.log(probability)
        return entropy

    def _entropy(self, probabilities: List[float]) -> float:
        entropy = 0.0
        for probability in probabilities:
            if probability > 0.0:
                entropy -= probability * math.log(probability)
        return entropy

    def _velocity_for_candidate(self, candidate: Dict[str, float]) -> Twist:
        cmd = Twist()
        if candidate["name"] == "stay":
            return cmd

        _, _, robot_yaw = self.robot_pose
        angle_error = normalize_angle(candidate["heading"] - robot_yaw)
        cmd.angular.z = clamp(
            self.angular_gain * angle_error,
            -self.max_angular_vel,
            self.max_angular_vel,
        )
        if abs(angle_error) < self.rotate_in_place_angle:
            cmd.linear.x = self.max_linear_vel
        return cmd

    def _smooth_velocity(self, cmd: Twist) -> Twist:
        alpha = self.velocity_smoothing_alpha
        smoothed = Twist()
        smoothed.linear.x = alpha * cmd.linear.x + (1.0 - alpha) * self.filtered_cmd.linear.x
        smoothed.angular.z = alpha * cmd.angular.z + (1.0 - alpha) * self.filtered_cmd.angular.z
        self.filtered_cmd = smoothed
        return smoothed

    def _best_goal_msg(self, candidate: Dict[str, float]) -> PoseStamped:
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = self.probability_grid.header.frame_id
        goal.pose.position.x = candidate["x"]
        goal.pose.position.y = candidate["y"]
        qx, qy, qz, qw = yaw_to_quaternion(candidate["heading"])
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw
        return goal

    def _goal_traversability(self, x: float, y: float) -> Tuple[bool, str]:
        if self.costmap is None:
            return True, "none"

        source_frame = self.probability_grid.header.frame_id or "map"
        transformed = self._transform_xy(
            x, y, source_frame, self.costmap.header.frame_id
        )
        if transformed is None:
            return False, "tf_unavailable"

        costmap_x, costmap_y = transformed
        center = self._world_to_grid(self.costmap, costmap_x, costmap_y)
        if center is None:
            return False, "outside_costmap"
        center_col, center_row = center
        radius_cells = max(0, int(math.ceil(self.candidate_radius / self.costmap.info.resolution)))

        for row in range(center_row - radius_cells, center_row + radius_cells + 1):
            for col in range(center_col - radius_cells, center_col + radius_cells + 1):
                if not self._grid_contains(self.costmap, col, row):
                    return False, "footprint_outside"
                wx, wy = self._grid_to_world(self.costmap, col, row)
                if math.hypot(wx - costmap_x, wy - costmap_y) > self.candidate_radius:
                    continue
                value = self.costmap.data[row * self.costmap.info.width + col]
                if value < 0 and self.reject_unknown_goals:
                    return False, "unknown"
                if value >= self.obstacle_lethal_threshold:
                    return False, "lethal_or_inscribed"
        return True, "none"

    def _transform_xy(
        self,
        x: float,
        y: float,
        source_frame: str,
        target_frame: str,
    ) -> Optional[Tuple[float, float]]:
        if not target_frame or source_frame == target_frame:
            return x, y

        cache_key = (source_frame, target_frame)
        if cache_key not in self.transform_cache:
            try:
                self.transform_cache[cache_key] = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=0.05),
                )
            except TransformException:
                self.transform_cache[cache_key] = None

        transform = self.transform_cache[cache_key]
        if transform is None:
            return None
        translation = transform.transform.translation
        yaw = quaternion_to_yaw(transform.transform.rotation)
        return (
            translation.x + math.cos(yaw) * x - math.sin(yaw) * y,
            translation.y + math.sin(yaw) * x + math.cos(yaw) * y,
        )

    def _cell_center(self, index: int) -> Tuple[float, float]:
        grid = self.probability_grid
        row = index // grid.info.width
        col = index % grid.info.width
        return self._grid_to_world(grid, col, row)

    def _world_to_grid(self, grid: OccupancyGrid, x: float, y: float) -> Optional[Tuple[int, int]]:
        resolution = grid.info.resolution
        dx = x - grid.info.origin.position.x
        dy = y - grid.info.origin.position.y
        origin_yaw = quaternion_to_yaw(grid.info.origin.orientation)
        local_x = math.cos(origin_yaw) * dx + math.sin(origin_yaw) * dy
        local_y = -math.sin(origin_yaw) * dx + math.cos(origin_yaw) * dy
        col = int(math.floor(local_x / resolution))
        row = int(math.floor(local_y / resolution))
        if not self._grid_contains(grid, col, row):
            return None
        return col, row

    def _grid_to_world(self, grid: OccupancyGrid, col: int, row: int) -> Tuple[float, float]:
        local_x = (col + 0.5) * grid.info.resolution
        local_y = (row + 0.5) * grid.info.resolution
        origin_yaw = quaternion_to_yaw(grid.info.origin.orientation)
        return (
            grid.info.origin.position.x
            + math.cos(origin_yaw) * local_x
            - math.sin(origin_yaw) * local_y,
            grid.info.origin.position.y
            + math.sin(origin_yaw) * local_x
            + math.cos(origin_yaw) * local_y,
        )

    def _grid_contains(self, grid: OccupancyGrid, col: int, row: int) -> bool:
        return 0 <= col < grid.info.width and 0 <= row < grid.info.height

    def _markers(
        self,
        scored: List[Dict[str, float]],
        best: Optional[Dict[str, float]],
    ) -> MarkerArray:
        markers = MarkerArray()
        if self.probability_grid is None or self.robot_pose is None:
            return markers

        stamp = self.get_clock().now().to_msg()
        frame_id = self.probability_grid.header.frame_id
        valid_scores = [item["delta_entropy"] for item in scored if item["valid"]]
        min_delta = min(valid_scores) if valid_scores else 0.0
        max_delta = max(valid_scores) if valid_scores else 1.0
        span = max(1e-9, max_delta - min_delta)

        for index, item in enumerate(scored):
            quality = 0.0
            if item["valid"]:
                quality = (item["delta_entropy"] - min_delta) / span

            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = frame_id
            marker.ns = "infotaxis"
            marker.id = index
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            is_best = item is best
            marker.scale.x = 0.05 if not is_best else 0.09
            marker.scale.y = 0.12 if not is_best else 0.18
            marker.scale.z = 0.12 if not is_best else 0.18
            marker.color = self._score_color(quality, item["valid"], is_best)

            start = Point()
            start.x, start.y, _ = self.robot_pose
            start.z = 0.25
            end = Point()
            end.x = item["x"]
            end.y = item["y"]
            end.z = 0.25
            marker.points = [start, end]
            markers.markers.append(marker)
        return markers

    def _score_color(self, quality: float, valid: bool, selected: bool) -> ColorRGBA:
        if not valid:
            return ColorRGBA(r=0.25, g=0.25, b=0.25, a=0.35)
        return ColorRGBA(
            r=1.0 - quality,
            g=quality,
            b=0.20 if selected else 0.65,
            a=0.95 if selected else 0.55,
        )

    def _publish_status(self, text: str):
        self.status_pub.publish(String(data=text))


def main(args=None):
    rclpy.init(args=args)
    node = Infotaxis()
    try:
        rclpy.spin(node)
    finally:
        node.velocity_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
