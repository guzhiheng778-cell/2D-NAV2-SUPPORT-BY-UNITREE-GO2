#!/usr/bin/env python3

import math
from typing import Dict, List, Optional, Tuple

import rclpy
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from std_msgs.msg import ColorRGBA, String
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


class LocalInfoGain(Node):
    def __init__(self):
        super().__init__("local_info_gain")

        self.declare_parameter("robot_pose_topic", "/amcl_pose")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("probability_grid_topic", "/probability_grid")
        self.declare_parameter("source_estimate_topic", "/source_estimate")
        self.declare_parameter("costmap_topic", "/local_costmap/costmap")
        self.declare_parameter("velocity_topic", "/v_info")
        self.declare_parameter("best_goal_topic", "/best_info_goal")
        self.declare_parameter("status_topic", "/local_info_gain/status")
        self.declare_parameter("marker_topic", "/local_info_gain/markers")
        self.declare_parameter("control_rate_hz", 2.0)
        self.declare_parameter("candidate_distance", 0.80)
        self.declare_parameter("candidate_radius", 0.45)
        self.declare_parameter("visited_radius", 0.65)
        self.declare_parameter("max_linear_vel", 0.18)
        self.declare_parameter("max_angular_vel", 0.45)
        self.declare_parameter("linear_gain", 0.40)
        self.declare_parameter("angular_gain", 1.20)
        self.declare_parameter("rotate_in_place_angle", 0.75)
        self.declare_parameter("w_probability", 1.6)
        self.declare_parameter("w_pmax", 2.0)
        self.declare_parameter("w_unknown", 0.35)
        self.declare_parameter("w_obstacle", 0.8)
        self.declare_parameter("w_revisit", 2.0)
        self.declare_parameter("w_turn", 0.3)
        self.declare_parameter("w_backtrack", 0.0)
        self.declare_parameter("w_reverse_penalty", 0.20)
        self.declare_parameter("w_stop_penalty", 0.05)
        self.declare_parameter("unvisited_weight", 0.7)
        self.declare_parameter("uncertainty_weight", 0.3)
        self.declare_parameter("obstacle_lethal_threshold", 85)
        self.declare_parameter("obstacle_unknown_cost", 0.35)
        self.declare_parameter("revisit_memory_sec", 20.0)
        self.declare_parameter("revisit_radius", 0.65)
        self.declare_parameter("revisit_ignore_recent_sec", 2.0)
        self.declare_parameter("selection_hold_sec", 1.50)
        self.declare_parameter("selection_hysteresis", 0.12)
        self.declare_parameter("velocity_smoothing_alpha", 0.25)
        self.declare_parameter("publish_zero_without_inputs", True)

        self.candidate_distance = float(self.get_parameter("candidate_distance").value)
        self.candidate_radius = float(self.get_parameter("candidate_radius").value)
        self.visited_radius = float(self.get_parameter("visited_radius").value)
        self.max_linear_vel = float(self.get_parameter("max_linear_vel").value)
        self.max_angular_vel = float(self.get_parameter("max_angular_vel").value)
        self.linear_gain = float(self.get_parameter("linear_gain").value)
        self.angular_gain = float(self.get_parameter("angular_gain").value)
        self.rotate_in_place_angle = float(self.get_parameter("rotate_in_place_angle").value)
        self.w_probability = float(self.get_parameter("w_probability").value)
        self.w_pmax = float(self.get_parameter("w_pmax").value)
        self.w_unknown = float(self.get_parameter("w_unknown").value)
        self.w_obstacle = float(self.get_parameter("w_obstacle").value)
        self.w_revisit = float(self.get_parameter("w_revisit").value)
        self.w_turn = float(self.get_parameter("w_turn").value)
        self.w_backtrack = float(self.get_parameter("w_backtrack").value)
        self.w_reverse_penalty = float(self.get_parameter("w_reverse_penalty").value)
        self.w_stop_penalty = float(self.get_parameter("w_stop_penalty").value)
        self.unvisited_weight = float(self.get_parameter("unvisited_weight").value)
        self.uncertainty_weight = float(self.get_parameter("uncertainty_weight").value)
        self.obstacle_lethal_threshold = int(self.get_parameter("obstacle_lethal_threshold").value)
        self.obstacle_unknown_cost = float(self.get_parameter("obstacle_unknown_cost").value)
        self.revisit_memory_sec = float(self.get_parameter("revisit_memory_sec").value)
        self.revisit_radius = float(self.get_parameter("revisit_radius").value)
        self.revisit_ignore_recent_sec = float(
            self.get_parameter("revisit_ignore_recent_sec").value
        )
        self.selection_hold_sec = float(self.get_parameter("selection_hold_sec").value)
        self.selection_hysteresis = float(self.get_parameter("selection_hysteresis").value)
        self.velocity_smoothing_alpha = clamp(
            float(self.get_parameter("velocity_smoothing_alpha").value), 0.0, 1.0
        )
        self.publish_zero_without_inputs = bool(
            self.get_parameter("publish_zero_without_inputs").value
        )

        self.robot_pose: Optional[Tuple[float, float, float]] = None
        self.has_map_pose = False
        self.probability_grid: Optional[OccupancyGrid] = None
        self.source_estimate: Optional[Tuple[float, float]] = None
        self.costmap: Optional[OccupancyGrid] = None
        self.visited_keys = set()
        self.recent_path: List[Tuple[object, float, float]] = []
        self.last_scores: List[Dict[str, float]] = []
        self.selected_name: Optional[str] = None
        self.last_selected_heading: Optional[float] = None
        self.last_selection_time = None
        self.filtered_cmd = Twist()

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
            PoseStamped,
            self.get_parameter("source_estimate_topic").value,
            self._source_estimate_callback,
            10,
        )
        self.create_subscription(
            OccupancyGrid,
            self.get_parameter("costmap_topic").value,
            self._costmap_callback,
            2,
        )

        rate = max(0.5, float(self.get_parameter("control_rate_hz").value))
        self.timer = self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            "Local information gain ready: publishing candidate velocity on /v_info"
        )

    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        self.has_map_pose = True
        self.robot_pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            quaternion_to_yaw(msg.pose.pose.orientation),
        )
        self._mark_visited(self.robot_pose[0], self.robot_pose[1])
        self._record_recent_pose(self.robot_pose[0], self.robot_pose[1])

    def _odom_callback(self, msg: Odometry):
        if self.has_map_pose:
            return
        self.robot_pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            quaternion_to_yaw(msg.pose.pose.orientation),
        )
        self._mark_visited(self.robot_pose[0], self.robot_pose[1])
        self._record_recent_pose(self.robot_pose[0], self.robot_pose[1])

    def _probability_grid_callback(self, msg: OccupancyGrid):
        self.probability_grid = msg

    def _source_estimate_callback(self, msg: PoseStamped):
        self.source_estimate = (msg.pose.position.x, msg.pose.position.y)

    def _costmap_callback(self, msg: OccupancyGrid):
        self.costmap = msg

    def _tick(self):
        if self.robot_pose is None or self.probability_grid is None:
            self._publish_status("waiting_for_pose_or_probability_grid")
            if self.publish_zero_without_inputs:
                self.velocity_pub.publish(Twist())
            return

        candidates = self._candidate_points()
        scored = [self._score_candidate(candidate) for candidate in candidates]
        best = self._select_candidate(scored)
        self.last_scores = scored

        cmd = self._smooth_velocity(self._velocity_for_candidate(best))
        self.velocity_pub.publish(cmd)
        self.best_goal_pub.publish(self._best_goal_msg(best))
        self.marker_pub.publish(self._markers(scored, best))
        self._publish_status(
            "best={name} score={score:.3f} prob={probability:.3f} "
            "pmax_attr={pmax_attraction:.3f} unknown={unknown:.3f} "
            "unvisited={unvisited:.3f} entropy={uncertainty:.3f} "
            "obs={obstacle:.3f} turn={turn:.3f} backtrack={backtrack:.3f} "
            "revisit={revisit:.3f}".format(**best)
        )

    def _select_candidate(self, scored: List[Dict[str, float]]) -> Dict[str, float]:
        best = max(scored, key=lambda item: item["score"])
        now = self.get_clock().now()

        previous = None
        if self.selected_name is not None:
            previous = next(
                (item for item in scored if item["name"] == self.selected_name),
                None,
            )

        if previous is None:
            self.selected_name = best["name"]
            self.last_selected_heading = best["heading"]
            self.last_selection_time = now
            return best

        hold_active = (
            self.last_selection_time is not None
            and (now - self.last_selection_time).nanoseconds * 1e-9 < self.selection_hold_sec
        )
        improvement = best["score"] - previous["score"]
        if best["name"] != previous["name"] and (
            hold_active or improvement < self.selection_hysteresis
        ):
            return previous

        if best["name"] != previous["name"]:
            self.selected_name = best["name"]
            self.last_selected_heading = best["heading"]
            self.last_selection_time = now
        elif self.last_selected_heading is None:
            self.last_selected_heading = best["heading"]
        return best

    def _candidate_points(self) -> List[Dict[str, float]]:
        robot_x, robot_y, robot_yaw = self.robot_pose
        definitions = [
            ("front", 0.0),
            ("left_front", math.pi / 4.0),
            ("right_front", -math.pi / 4.0),
            ("left", math.pi / 2.0),
            ("right", -math.pi / 2.0),
            ("back", math.pi),
            ("stay", None),
        ]

        candidates = []
        for name, relative_yaw in definitions:
            if relative_yaw is None:
                x, y = robot_x, robot_y
                heading = robot_yaw
            else:
                heading = robot_yaw + relative_yaw
                x = robot_x + self.candidate_distance * math.cos(heading)
                y = robot_y + self.candidate_distance * math.sin(heading)
            candidates.append(
                {
                    "name": name,
                    "x": x,
                    "y": y,
                    "heading": heading,
                    "relative_yaw": 0.0 if relative_yaw is None else relative_yaw,
                }
            )
        return candidates

    def _score_candidate(self, candidate: Dict[str, float]) -> Dict[str, float]:
        probability = self._mean_grid_value(
            self.probability_grid, candidate["x"], candidate["y"], self.candidate_radius
        )
        pmax_attraction = self._pmax_attraction(candidate["x"], candidate["y"])
        unvisited = self._unvisited_gain(candidate["x"], candidate["y"])
        uncertainty = self._local_entropy_gain(candidate["x"], candidate["y"])
        unknown = (
            self.unvisited_weight * unvisited
            + self.uncertainty_weight * uncertainty
        )
        obstacle = self._obstacle_cost(candidate["x"], candidate["y"])
        revisit = self._revisit_penalty(candidate["x"], candidate["y"])
        turn = self._turn_penalty(candidate["heading"])
        backtrack = self._backtrack_penalty(candidate["heading"])
        penalty = self._motion_penalty(candidate["name"])
        score = (
            self.w_probability * probability
            + self.w_pmax * pmax_attraction
            + self.w_unknown * unknown
            - self.w_obstacle * obstacle
            - self.w_revisit * revisit
            - self.w_turn * turn
            - self.w_backtrack * backtrack
            - penalty
        )

        result = dict(candidate)
        result.update(
            {
                "probability": probability,
                "pmax_attraction": pmax_attraction,
                "unknown": unknown,
                "unvisited": unvisited,
                "uncertainty": uncertainty,
                "obstacle": obstacle,
                "revisit": revisit,
                "turn": turn,
                "backtrack": backtrack,
                "penalty": penalty,
                "score": score,
            }
        )
        return result

    def _pmax_attraction(self, candidate_x: float, candidate_y: float) -> float:
        if self.source_estimate is None or self.robot_pose is None:
            return 0.0
        robot_x, robot_y, _ = self.robot_pose
        pmax_x, pmax_y = self.source_estimate
        dist_now = math.hypot(robot_x - pmax_x, robot_y - pmax_y)
        dist_candidate = math.hypot(candidate_x - pmax_x, candidate_y - pmax_y)
        scale = max(0.1, self.candidate_distance)
        return clamp((dist_now - dist_candidate) / scale, -1.0, 1.0)

    def _mean_grid_value(
        self, grid: Optional[OccupancyGrid], x: float, y: float, radius: float
    ) -> float:
        if grid is None:
            return 0.0

        center = self._world_to_grid(grid, x, y)
        if center is None:
            return 0.0
        center_col, center_row = center
        radius_cells = max(1, int(math.ceil(radius / grid.info.resolution)))

        total = 0.0
        count = 0
        for row in range(center_row - radius_cells, center_row + radius_cells + 1):
            for col in range(center_col - radius_cells, center_col + radius_cells + 1):
                if not self._grid_contains(grid, col, row):
                    continue
                wx, wy = self._grid_to_world(grid, col, row)
                if math.hypot(wx - x, wy - y) > radius:
                    continue
                value = grid.data[row * grid.info.width + col]
                if value < 0:
                    continue
                total += clamp(value / 100.0, 0.0, 1.0)
                count += 1

        if count == 0:
            return 0.0
        return total / count

    def _unvisited_gain(self, x: float, y: float) -> float:
        grid = self.probability_grid
        if grid is None:
            return 0.0

        center = self._world_to_grid(grid, x, y)
        if center is None:
            return 0.0
        center_col, center_row = center
        radius_cells = max(1, int(math.ceil(self.candidate_radius / grid.info.resolution)))

        total = 0
        unseen = 0
        for row in range(center_row - radius_cells, center_row + radius_cells + 1):
            for col in range(center_col - radius_cells, center_col + radius_cells + 1):
                if not self._grid_contains(grid, col, row):
                    continue
                wx, wy = self._grid_to_world(grid, col, row)
                if math.hypot(wx - x, wy - y) > self.candidate_radius:
                    continue
                total += 1
                if self._visited_distance(wx, wy) > self.visited_radius:
                    unseen += 1

        if total == 0:
            return 0.0
        return unseen / total

    def _local_entropy_gain(self, x: float, y: float) -> float:
        grid = self.probability_grid
        if grid is None:
            return 0.0

        center = self._world_to_grid(grid, x, y)
        if center is None:
            return 0.0
        center_col, center_row = center
        radius_cells = max(1, int(math.ceil(self.candidate_radius / grid.info.resolution)))

        total_entropy = 0.0
        count = 0
        for row in range(center_row - radius_cells, center_row + radius_cells + 1):
            for col in range(center_col - radius_cells, center_col + radius_cells + 1):
                if not self._grid_contains(grid, col, row):
                    continue
                wx, wy = self._grid_to_world(grid, col, row)
                if math.hypot(wx - x, wy - y) > self.candidate_radius:
                    continue
                value = grid.data[row * grid.info.width + col]
                if value < 0:
                    continue
                p = clamp(value / 100.0, 1e-6, 1.0 - 1e-6)
                total_entropy += -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)
                count += 1

        if count == 0:
            return 0.0
        return clamp((total_entropy / count) / math.log(2.0), 0.0, 1.0)

    def _obstacle_cost(self, x: float, y: float) -> float:
        if self.costmap is None:
            return 0.0

        center = self._world_to_grid(self.costmap, x, y)
        if center is None:
            return 1.0
        center_col, center_row = center
        radius_cells = max(1, int(math.ceil(self.candidate_radius / self.costmap.info.resolution)))

        max_cost = 0.0
        for row in range(center_row - radius_cells, center_row + radius_cells + 1):
            for col in range(center_col - radius_cells, center_col + radius_cells + 1):
                if not self._grid_contains(self.costmap, col, row):
                    max_cost = max(max_cost, 1.0)
                    continue
                wx, wy = self._grid_to_world(self.costmap, col, row)
                if math.hypot(wx - x, wy - y) > self.candidate_radius:
                    continue
                raw = self.costmap.data[row * self.costmap.info.width + col]
                if raw < 0:
                    cost = self.obstacle_unknown_cost
                else:
                    cost = clamp(raw / 100.0, 0.0, 1.0)
                    if raw >= self.obstacle_lethal_threshold:
                        cost = 1.0
                max_cost = max(max_cost, cost)
        return max_cost

    def _motion_penalty(self, name: str) -> float:
        if name == "back":
            return self.w_reverse_penalty
        if name == "stay":
            return self.w_stop_penalty
        return 0.0

    def _revisit_penalty(self, x: float, y: float) -> float:
        if self.robot_pose is None:
            return 0.0
        self._prune_recent_path()
        robot_x, robot_y, _ = self.robot_pose

        best = 0.0
        for stamp, recent_x, recent_y in self.recent_path:
            age = self._stamp_age_sec(stamp)
            if age < self.revisit_ignore_recent_sec:
                continue
            distance = self._distance_point_to_segment(
                recent_x,
                recent_y,
                robot_x,
                robot_y,
                x,
                y,
            )
            if distance >= self.revisit_radius:
                continue
            proximity = 1.0 - distance / max(1e-6, self.revisit_radius)
            active_window = max(1e-6, self.revisit_memory_sec - self.revisit_ignore_recent_sec)
            age_weight = 1.0 - (age - self.revisit_ignore_recent_sec) / active_window
            age_weight = clamp(age_weight, 0.20, 1.0)
            best = max(best, proximity * age_weight)
        return clamp(best, 0.0, 1.0)

    def _turn_penalty(self, heading: float) -> float:
        if self.last_selected_heading is None:
            return 0.0
        return abs(normalize_angle(heading - self.last_selected_heading))

    def _backtrack_penalty(self, heading: float) -> float:
        if self.last_selected_heading is None:
            return 0.0
        turn = abs(normalize_angle(heading - self.last_selected_heading))
        start_penalizing = 0.60 * math.pi
        if turn <= start_penalizing:
            return 0.0
        return clamp((turn - start_penalizing) / (math.pi - start_penalizing), 0.0, 1.0)

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
        else:
            cmd.linear.x = 0.0
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

    def _mark_visited(self, x: float, y: float):
        if self.probability_grid is None:
            return
        index = self._world_to_grid(self.probability_grid, x, y)
        if index is not None:
            self.visited_keys.add(index)

    def _record_recent_pose(self, x: float, y: float):
        self.recent_path.append((self.get_clock().now(), x, y))
        self._prune_recent_path()

    def _prune_recent_path(self):
        now = self.get_clock().now()
        self.recent_path = [
            item for item in self.recent_path
            if (now - item[0]).nanoseconds * 1e-9 <= self.revisit_memory_sec
        ]

    def _stamp_age_sec(self, stamp) -> float:
        return (self.get_clock().now() - stamp).nanoseconds * 1e-9

    def _distance_point_to_segment(
        self,
        px: float,
        py: float,
        ax: float,
        ay: float,
        bx: float,
        by: float,
    ) -> float:
        abx = bx - ax
        aby = by - ay
        denom = abx * abx + aby * aby
        if denom <= 1e-9:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * abx + (py - ay) * aby) / denom
        t = clamp(t, 0.0, 1.0)
        closest_x = ax + t * abx
        closest_y = ay + t * aby
        return math.hypot(px - closest_x, py - closest_y)

    def _visited_distance(self, x: float, y: float) -> float:
        if not self.visited_keys or self.probability_grid is None:
            return float("inf")
        best = float("inf")
        for col, row in self.visited_keys:
            wx, wy = self._grid_to_world(self.probability_grid, col, row)
            best = min(best, math.hypot(wx - x, wy - y))
            if best <= self.visited_radius:
                break
        return best

    def _world_to_grid(self, grid: OccupancyGrid, x: float, y: float) -> Optional[Tuple[int, int]]:
        resolution = grid.info.resolution
        col = int(math.floor((x - grid.info.origin.position.x) / resolution))
        row = int(math.floor((y - grid.info.origin.position.y) / resolution))
        if not self._grid_contains(grid, col, row):
            return None
        return col, row

    def _grid_to_world(self, grid: OccupancyGrid, col: int, row: int) -> Tuple[float, float]:
        x = grid.info.origin.position.x + (col + 0.5) * grid.info.resolution
        y = grid.info.origin.position.y + (row + 0.5) * grid.info.resolution
        return x, y

    def _grid_contains(self, grid: OccupancyGrid, col: int, row: int) -> bool:
        return 0 <= col < grid.info.width and 0 <= row < grid.info.height

    def _markers(self, scored: List[Dict[str, float]], best: Dict[str, float]) -> MarkerArray:
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        frame_id = self.probability_grid.header.frame_id

        min_score = min(item["score"] for item in scored)
        max_score = max(item["score"] for item in scored)
        span = max(1e-6, max_score - min_score)

        for index, item in enumerate(scored):
            quality = (item["score"] - min_score) / span
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = frame_id
            marker.ns = "local_info_gain"
            marker.id = index
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.05 if item["name"] != best["name"] else 0.09
            marker.scale.y = 0.12 if item["name"] != best["name"] else 0.18
            marker.scale.z = 0.12 if item["name"] != best["name"] else 0.18
            marker.color = self._score_color(quality, item["name"] == best["name"])

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

    def _score_color(self, quality: float, selected: bool) -> ColorRGBA:
        color = ColorRGBA()
        color.r = 1.0 - quality
        color.g = quality
        color.b = 0.15 if selected else 0.55
        color.a = 0.95 if selected else 0.55
        return color

    def _publish_status(self, text: str):
        self.status_pub.publish(String(data=text))


def main(args=None):
    rclpy.init(args=args)
    node = LocalInfoGain()
    try:
        rclpy.spin(node)
    finally:
        node.velocity_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
