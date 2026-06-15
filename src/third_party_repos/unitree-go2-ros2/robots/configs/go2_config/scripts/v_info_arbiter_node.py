#!/usr/bin/env python3

import math
from enum import Enum
from typing import Optional, Tuple

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


class ControlState(Enum):
    PATROL = "PATROL"
    SOURCE_SEEK = "SOURCE_SEEK"
    RECOVERY = "RECOVERY"


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def quaternion_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw: float):
    half_yaw = 0.5 * yaw
    return 0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)


class VInfoArbiter(Node):
    def __init__(self):
        super().__init__("v_info_arbiter")

        self.declare_parameter("v_info_topic", "/v_info")
        self.declare_parameter("best_info_goal_topic", "/best_info_goal")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("pmax_topic", "/pmax")
        self.declare_parameter("entropy_topic", "/entropy")
        self.declare_parameter("source_estimate_topic", "/source_estimate")
        self.declare_parameter("odor_detection_topic", "/odor_detection")
        self.declare_parameter("costmap_topic", "/local_costmap/costmap")
        self.declare_parameter("robot_pose_topic", "/amcl_pose")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("active_topic", "/source_seek/active")
        self.declare_parameter("success_topic", "/source_seek/succeeded")
        self.declare_parameter("state_topic", "/v_info_arbiter/state")
        self.declare_parameter("status_topic", "/v_info_arbiter/status")
        self.declare_parameter("goal_frame", "map")

        self.declare_parameter("control_rate_hz", 10.0)
        self.declare_parameter("pmax_enter_threshold", 0.20)
        self.declare_parameter("pmax_exit_threshold", 0.15)
        self.declare_parameter("pmax_drop_tolerance", 0.05)
        self.declare_parameter("v_info_timeout_sec", 0.60)
        self.declare_parameter("best_info_goal_timeout_sec", 2.0)
        self.declare_parameter("hit_timeout_sec", 8.0)
        self.declare_parameter("max_source_seek_duration_sec", 15.0)
        self.declare_parameter("max_source_seek_extended_duration_sec", 90.0)
        self.declare_parameter("duration_reward_per_local_goal_sec", 8.0)
        self.declare_parameter("max_source_seek_distance", 2.00)
        self.declare_parameter("source_reached_distance", 0.45)
        self.declare_parameter("source_success_min_duration_sec", 2.0)
        self.declare_parameter("pmax_success_threshold", 0.0020)
        self.declare_parameter("recovery_stop_sec", 1.0)
        self.declare_parameter("patrol_resume_lockout_sec", 3.0)
        self.declare_parameter("entropy_grace_sec", 2.0)
        self.declare_parameter("min_entropy_drop", 0.01)
        self.declare_parameter("max_linear_vel", 0.16)
        self.declare_parameter("max_angular_vel", 0.40)
        self.declare_parameter("local_goal_distance", 0.80)
        self.declare_parameter("local_goal_timeout_sec", 12.0)
        self.declare_parameter("local_goal_tolerance", 0.30)
        self.declare_parameter("goal_cost_threshold", 100)
        self.declare_parameter("goal_search_radius", 0.0)
        self.declare_parameter("reject_unknown_goals", False)
        self.declare_parameter("no_valid_goal_retry_sec", 0.50)
        self.declare_parameter("max_no_valid_goal_attempts", 4)
        self.declare_parameter("source_seek_failure_cooldown_sec", 10.0)
        self.declare_parameter("allow_reverse", False)
        self.declare_parameter("require_nonzero_v_info_on_enter", True)
        self.declare_parameter("min_enter_speed", 0.02)

        self.control_rate_hz = max(1.0, float(self.get_parameter("control_rate_hz").value))
        self.pmax_enter_threshold = float(self.get_parameter("pmax_enter_threshold").value)
        self.pmax_exit_threshold = float(self.get_parameter("pmax_exit_threshold").value)
        self.pmax_drop_tolerance = float(self.get_parameter("pmax_drop_tolerance").value)
        self.goal_frame = str(self.get_parameter("goal_frame").value)
        self.v_info_timeout_sec = float(self.get_parameter("v_info_timeout_sec").value)
        self.best_info_goal_timeout_sec = float(
            self.get_parameter("best_info_goal_timeout_sec").value
        )
        self.hit_timeout_sec = float(self.get_parameter("hit_timeout_sec").value)
        self.max_source_seek_duration_sec = float(
            self.get_parameter("max_source_seek_duration_sec").value
        )
        self.max_source_seek_extended_duration_sec = float(
            self.get_parameter("max_source_seek_extended_duration_sec").value
        )
        self.duration_reward_per_local_goal_sec = float(
            self.get_parameter("duration_reward_per_local_goal_sec").value
        )
        self.max_source_seek_distance = float(self.get_parameter("max_source_seek_distance").value)
        self.source_reached_distance = float(self.get_parameter("source_reached_distance").value)
        self.source_success_min_duration_sec = float(
            self.get_parameter("source_success_min_duration_sec").value
        )
        self.pmax_success_threshold = float(self.get_parameter("pmax_success_threshold").value)
        self.recovery_stop_sec = float(self.get_parameter("recovery_stop_sec").value)
        self.patrol_resume_lockout_sec = float(
            self.get_parameter("patrol_resume_lockout_sec").value
        )
        self.entropy_grace_sec = float(self.get_parameter("entropy_grace_sec").value)
        self.min_entropy_drop = float(self.get_parameter("min_entropy_drop").value)
        self.max_linear_vel = float(self.get_parameter("max_linear_vel").value)
        self.max_angular_vel = float(self.get_parameter("max_angular_vel").value)
        self.local_goal_distance = float(self.get_parameter("local_goal_distance").value)
        self.local_goal_timeout_sec = float(self.get_parameter("local_goal_timeout_sec").value)
        self.local_goal_tolerance = float(self.get_parameter("local_goal_tolerance").value)
        self.goal_cost_threshold = int(self.get_parameter("goal_cost_threshold").value)
        self.goal_search_radius = float(self.get_parameter("goal_search_radius").value)
        self.reject_unknown_goals = bool(self.get_parameter("reject_unknown_goals").value)
        self.no_valid_goal_retry_sec = float(self.get_parameter("no_valid_goal_retry_sec").value)
        self.max_no_valid_goal_attempts = int(
            self.get_parameter("max_no_valid_goal_attempts").value
        )
        self.source_seek_failure_cooldown_sec = float(
            self.get_parameter("source_seek_failure_cooldown_sec").value
        )
        self.allow_reverse = bool(self.get_parameter("allow_reverse").value)
        self.require_nonzero_v_info_on_enter = bool(
            self.get_parameter("require_nonzero_v_info_on_enter").value
        )
        self.min_enter_speed = float(self.get_parameter("min_enter_speed").value)

        self.state = ControlState.PATROL
        self.state_enter_time = self.get_clock().now()
        self.last_v_info: Optional[Twist] = None
        self.last_v_info_time = None
        self.last_best_info_goal: Optional[PoseStamped] = None
        self.last_best_info_goal_time = None
        self.pmax = 0.0
        self.source_seek_peak_pmax = 0.0
        self.entropy: Optional[float] = None
        self.source_seek_start_entropy: Optional[float] = None
        self.source_seek_duration_bonus_sec = 0.0
        self.source_seek_successful_local_goals = 0
        self.last_hit_time = None
        self.costmap: Optional[OccupancyGrid] = None
        self.robot_pose: Optional[Tuple[float, float, float]] = None
        self.source_estimate_xy: Optional[Tuple[float, float]] = None
        self.has_map_pose = False
        self.source_seek_start_xy: Optional[Tuple[float, float]] = None
        self.patrol_resume_until = None
        self.local_goal_handle = None
        self.local_goal_in_progress = False
        self.local_goal_sent_time = None
        self.local_goal_xy: Optional[Tuple[float, float]] = None
        self.local_goal_result_status: Optional[int] = None
        self.last_no_valid_goal_time = None
        self.no_valid_goal_attempts = 0
        self.source_seek_cooldown_until = None
        self.source_reached_latched = False

        self.cmd_pub = self.create_publisher(Twist, self.get_parameter("cmd_vel_topic").value, 10)
        self.active_pub = self.create_publisher(
            Bool, self.get_parameter("active_topic").value, 10
        )
        self.success_pub = self.create_publisher(
            Bool, self.get_parameter("success_topic").value, 10
        )
        self.state_pub = self.create_publisher(
            String, self.get_parameter("state_topic").value, 10
        )
        self.status_pub = self.create_publisher(
            String, self.get_parameter("status_topic").value, 10
        )
        self.action_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        self.create_subscription(Twist, self.get_parameter("v_info_topic").value, self._v_info_cb, 10)
        self.create_subscription(
            PoseStamped,
            self.get_parameter("best_info_goal_topic").value,
            self._best_info_goal_cb,
            10,
        )
        self.create_subscription(Float32, self.get_parameter("pmax_topic").value, self._pmax_cb, 10)
        self.create_subscription(
            Float32, self.get_parameter("entropy_topic").value, self._entropy_cb, 10
        )
        self.create_subscription(
            PoseStamped,
            self.get_parameter("source_estimate_topic").value,
            self._source_estimate_cb,
            10,
        )
        self.create_subscription(
            Bool, self.get_parameter("odor_detection_topic").value, self._detection_cb, 20
        )
        self.create_subscription(
            OccupancyGrid, self.get_parameter("costmap_topic").value, self._costmap_cb, 2
        )
        self.create_subscription(
            PoseWithCovarianceStamped,
            self.get_parameter("robot_pose_topic").value,
            self._pose_cb,
            10,
        )
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self._odom_cb, 20)

        self.timer = self.create_timer(1.0 / self.control_rate_hz, self._tick)
        self.get_logger().info(
            "v_info arbiter ready: PATROL -> SOURCE_SEEK when "
            f"Pmax>{self.pmax_enter_threshold:.3f} and /v_info is safe"
        )

    def _v_info_cb(self, msg: Twist):
        self.last_v_info = msg
        self.last_v_info_time = self.get_clock().now()

    def _best_info_goal_cb(self, msg: PoseStamped):
        self.last_best_info_goal = msg
        self.last_best_info_goal_time = self.get_clock().now()

    def _pmax_cb(self, msg: Float32):
        self.pmax = float(msg.data)

    def _entropy_cb(self, msg: Float32):
        self.entropy = float(msg.data)

    def _source_estimate_cb(self, msg: PoseStamped):
        self.source_estimate_xy = (msg.pose.position.x, msg.pose.position.y)

    def _detection_cb(self, msg: Bool):
        if msg.data:
            self.last_hit_time = self.get_clock().now()

    def _costmap_cb(self, msg: OccupancyGrid):
        self.costmap = msg

    def _pose_cb(self, msg: PoseWithCovarianceStamped):
        self.has_map_pose = True
        self.robot_pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            quaternion_to_yaw(msg.pose.pose.orientation),
        )

    def _odom_cb(self, msg: Odometry):
        if self.has_map_pose:
            return
        self.robot_pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            quaternion_to_yaw(msg.pose.pose.orientation),
        )

    def _tick(self):
        if self.state == ControlState.PATROL:
            self._tick_patrol()
        elif self.state == ControlState.SOURCE_SEEK:
            self._tick_source_seek()
        else:
            self._tick_recovery()

        self.active_pub.publish(Bool(data=self.state != ControlState.PATROL))
        self.success_pub.publish(Bool(data=self.source_reached_latched))
        self.state_pub.publish(String(data=self.state.value))

    def _tick_patrol(self):
        if self._can_enter_source_seek():
            self._enter_state(ControlState.SOURCE_SEEK, "pmax_high_and_v_info_safe")
            self.cmd_pub.publish(Twist())
            return
        self._publish_status(f"PATROL nav2_active blocked_by={self._entry_block_reason()}")

    def _tick_source_seek(self):
        exit_reason = self._source_seek_exit_reason()
        if exit_reason:
            self._enter_state(ControlState.RECOVERY, exit_reason)
            self.cmd_pub.publish(Twist())
            return

        self.source_seek_peak_pmax = max(self.source_seek_peak_pmax, self.pmax)

        if self.local_goal_in_progress:
            distance = self._distance_to_local_goal()
            self._publish_status(
                f"SOURCE_SEEK nav2_goal_active pmax={self.pmax:.3f} "
                f"goal_dist={distance:.2f}"
            )
            return

        if self.local_goal_result_status is not None:
            status = self.local_goal_result_status
            self.local_goal_result_status = None
            if status == GoalStatus.STATUS_SUCCEEDED:
                self._reward_source_seek_duration()
                self._publish_status(
                    "SOURCE_SEEK local_goal_reached recalculating "
                    f"duration_limit={self._effective_source_seek_duration_sec():.1f}s "
                    f"bonus={self.source_seek_duration_bonus_sec:.1f}s"
                )
            else:
                self._enter_state(ControlState.RECOVERY, f"local_goal_failed_status_{status}")
                self.cmd_pub.publish(Twist())
                return

        if self._waiting_to_retry_local_goal():
            self._publish_status("SOURCE_SEEK waiting_to_retry_local_goal")
            return

        goal = self._build_local_goal()
        if goal is None:
            self.no_valid_goal_attempts += 1
            if self.no_valid_goal_attempts >= self.max_no_valid_goal_attempts:
                self._enter_state(
                    ControlState.RECOVERY,
                    f"no_valid_local_nav2_goal_limit_{self.no_valid_goal_attempts}",
                )
                self.cmd_pub.publish(Twist())
                return
            self.last_no_valid_goal_time = self.get_clock().now()
            self._publish_status(
                "SOURCE_SEEK no_valid_local_nav2_goal "
                f"attempt={self.no_valid_goal_attempts}/"
                f"{self.max_no_valid_goal_attempts} "
                "retrying_with_shorter_or_side_candidates"
            )
            return

        self._send_local_goal(goal)

    def _tick_recovery(self):
        self.cmd_pub.publish(Twist())
        if self._state_age_sec() >= self.recovery_stop_sec:
            self._enter_state(ControlState.PATROL, "recovery_complete")
            return
        self._publish_status("RECOVERY stopping_before_patrol_resume")

    def _can_enter_source_seek(self) -> bool:
        return self._entry_block_reason() == "none"

    def _entry_block_reason(self) -> str:
        if self._patrol_resume_locked():
            remaining = (self.patrol_resume_until - self.get_clock().now()).nanoseconds * 1e-9
            return f"patrol_resume_lockout({max(0.0, remaining):.2f}s)"
        if self._source_seek_cooldown_locked():
            remaining = (self.source_seek_cooldown_until - self.get_clock().now()).nanoseconds * 1e-9
            return f"source_seek_cooldown({max(0.0, remaining):.2f}s)"
        if self.pmax <= self.pmax_enter_threshold:
            return f"pmax_low({self.pmax:.4f}<={self.pmax_enter_threshold:.4f})"
        has_fresh_best_goal = self._best_info_goal_is_fresh()
        if not self._v_info_is_fresh() and not has_fresh_best_goal:
            return "info_goal_stale"
        if (
            not has_fresh_best_goal
            and self.require_nonzero_v_info_on_enter
            and not self._v_info_has_motion()
        ):
            return "v_info_zero"
        return "none"

    def _source_seek_exit_reason(self) -> str:
        if self._source_reached():
            return "source_reached"
        if self.pmax < self.pmax_exit_threshold:
            return "pmax_below_exit_threshold"
        if self.source_seek_peak_pmax - self.pmax > self.pmax_drop_tolerance:
            return "pmax_dropped"
        if not self._v_info_is_fresh() and not self._best_info_goal_is_fresh():
            return "info_goal_timeout"
        if self._no_hit_too_long():
            return "hit_timeout"
        if self._state_age_sec() > self._effective_source_seek_duration_sec():
            return (
                "source_seek_duration_timeout"
                f"({self._state_age_sec():.1f}>"
                f"{self._effective_source_seek_duration_sec():.1f}s)"
            )
        if self._source_seek_distance() > self.max_source_seek_distance:
            return "source_seek_distance_limit"
        if self._local_goal_timed_out():
            return "local_goal_timeout"
        if self._entropy_not_improving():
            return "entropy_not_decreasing"
        return ""

    def _limited_v_info(self) -> Twist:
        cmd = Twist()
        if self.last_v_info is None:
            return cmd

        lower_linear = -self.max_linear_vel if self.allow_reverse else 0.0
        cmd.linear.x = clamp(self.last_v_info.linear.x, lower_linear, self.max_linear_vel)
        cmd.angular.z = clamp(
            self.last_v_info.angular.z, -self.max_angular_vel, self.max_angular_vel
        )
        return cmd

    def _build_local_goal(self) -> Optional[PoseStamped]:
        if self.robot_pose is None:
            return None
        if not self.action_client.wait_for_server(timeout_sec=0.1):
            self._publish_status("SOURCE_SEEK waiting_for_nav2_action_server")
            return None

        direct_goal = self._build_goal_from_best_info_goal()
        if direct_goal is not None:
            return direct_goal

        if self.last_v_info is None:
            return None

        robot_x, robot_y, robot_yaw = self.robot_pose
        direction_yaw = self._v_info_direction_yaw(robot_yaw)
        candidate = self._first_traversable_goal(robot_x, robot_y, direction_yaw)
        if candidate is None:
            return None
        goal_x, goal_y, goal_yaw = candidate

        goal = PoseStamped()
        goal.header.frame_id = self.goal_frame
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = goal_x
        goal.pose.position.y = goal_y
        qx, qy, qz, qw = yaw_to_quaternion(goal_yaw)
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw
        return goal

    def _build_goal_from_best_info_goal(self) -> Optional[PoseStamped]:
        if not self._best_info_goal_is_fresh() or self.last_best_info_goal is None:
            return None

        source = self.last_best_info_goal
        x = source.pose.position.x
        y = source.pose.position.y
        if not self._goal_is_traversable(x, y):
            return None

        goal = PoseStamped()
        goal.header.frame_id = source.header.frame_id or self.goal_frame
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose = source.pose
        return goal

    def _v_info_direction_yaw(self, robot_yaw: float) -> float:
        linear = self.last_v_info.linear.x
        angular = self.last_v_info.angular.z
        if abs(linear) >= self.min_enter_speed:
            lookahead_time = 1.2
            return robot_yaw + clamp(angular, -self.max_angular_vel, self.max_angular_vel) * lookahead_time
        if abs(angular) >= self.min_enter_speed:
            return robot_yaw + math.copysign(math.pi / 2.0, angular)
        return robot_yaw

    def _first_traversable_goal(
        self, robot_x: float, robot_y: float, direction_yaw: float
    ) -> Optional[Tuple[float, float, float]]:
        distances = [
            self.local_goal_distance,
            0.75 * self.local_goal_distance,
            0.50 * self.local_goal_distance,
            0.35 * self.local_goal_distance,
            0.25 * self.local_goal_distance,
        ]
        yaw_offsets = [
            0.0,
            0.30,
            -0.30,
            0.60,
            -0.60,
            math.pi / 2.0,
            -math.pi / 2.0,
            math.pi,
        ]
        for distance in distances:
            for yaw_offset in yaw_offsets:
                yaw = direction_yaw + yaw_offset
                x = robot_x + distance * math.cos(yaw)
                y = robot_y + distance * math.sin(yaw)
                if self._goal_is_traversable(x, y):
                    return x, y, yaw
        return None

    def _goal_is_traversable(self, x: float, y: float) -> bool:
        if self.costmap is None:
            return True
        center = self._world_to_grid(self.costmap, x, y)
        if center is None:
            return False
        center_col, center_row = center
        radius_cells = max(0, int(math.ceil(self.goal_search_radius / self.costmap.info.resolution)))
        for row in range(center_row - radius_cells, center_row + radius_cells + 1):
            for col in range(center_col - radius_cells, center_col + radius_cells + 1):
                if not self._grid_contains(self.costmap, col, row):
                    return False
                wx, wy = self._grid_to_world(self.costmap, col, row)
                if self.goal_search_radius > 0.0 and math.hypot(wx - x, wy - y) > self.goal_search_radius:
                    continue
                value = self.costmap.data[row * self.costmap.info.width + col]
                if value < 0 and self.reject_unknown_goals:
                    return False
                if value >= self.goal_cost_threshold:
                    return False
        return True

    def _send_local_goal(self, pose: PoseStamped):
        goal = NavigateToPose.Goal()
        goal.pose = pose
        self.local_goal_in_progress = True
        self.local_goal_sent_time = self.get_clock().now()
        self.local_goal_xy = (pose.pose.position.x, pose.pose.position.y)
        self.local_goal_result_status = None
        self.no_valid_goal_attempts = 0
        self._publish_status(
            f"SOURCE_SEEK sending_local_nav2_goal x={pose.pose.position.x:.2f} "
            f"y={pose.pose.position.y:.2f}"
        )
        future = self.action_client.send_goal_async(goal)
        future.add_done_callback(self._local_goal_response_cb)

    def _local_goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Local v_info Nav2 goal was rejected")
            self.local_goal_in_progress = False
            self.local_goal_handle = None
            self.local_goal_result_status = GoalStatus.STATUS_ABORTED
            return
        self.local_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._local_goal_result_cb)

    def _local_goal_result_cb(self, future):
        result = future.result()
        self.local_goal_result_status = result.status
        self.local_goal_in_progress = False
        self.local_goal_handle = None

    def _cancel_local_goal(self):
        if self.local_goal_handle is not None:
            self.local_goal_handle.cancel_goal_async()
        self.local_goal_in_progress = False
        self.local_goal_handle = None
        self.local_goal_xy = None
        self.local_goal_sent_time = None
        self.local_goal_result_status = None

    def _local_goal_timed_out(self) -> bool:
        if not self.local_goal_in_progress or self.local_goal_sent_time is None:
            return False
        return self._age_sec(self.local_goal_sent_time) >= self.local_goal_timeout_sec

    def _waiting_to_retry_local_goal(self) -> bool:
        if self.last_no_valid_goal_time is None:
            return False
        if self._age_sec(self.last_no_valid_goal_time) >= self.no_valid_goal_retry_sec:
            self.last_no_valid_goal_time = None
            return False
        return True

    def _distance_to_local_goal(self) -> float:
        if self.local_goal_xy is None or self.robot_pose is None:
            return 0.0
        return math.hypot(
            self.robot_pose[0] - self.local_goal_xy[0],
            self.robot_pose[1] - self.local_goal_xy[1],
        )

    def _v_info_is_fresh(self) -> bool:
        if self.last_v_info is None or self.last_v_info_time is None:
            return False
        return self._age_sec(self.last_v_info_time) <= self.v_info_timeout_sec

    def _best_info_goal_is_fresh(self) -> bool:
        if self.last_best_info_goal is None or self.last_best_info_goal_time is None:
            return False
        return self._age_sec(self.last_best_info_goal_time) <= self.best_info_goal_timeout_sec

    def _v_info_has_motion(self) -> bool:
        if self.last_v_info is None:
            return False
        linear = abs(self.last_v_info.linear.x)
        angular = abs(self.last_v_info.angular.z)
        return max(linear, angular) >= self.min_enter_speed

    def _no_hit_too_long(self) -> bool:
        if self.hit_timeout_sec <= 0.0:
            return False
        if self.last_hit_time is None:
            return self._state_age_sec() > self.hit_timeout_sec
        return self._age_sec(self.last_hit_time) > self.hit_timeout_sec

    def _source_seek_distance(self) -> float:
        if self.source_seek_start_xy is None or self.robot_pose is None:
            return 0.0
        dx = self.robot_pose[0] - self.source_seek_start_xy[0]
        dy = self.robot_pose[1] - self.source_seek_start_xy[1]
        return math.hypot(dx, dy)

    def _effective_source_seek_duration_sec(self) -> float:
        return min(
            self.max_source_seek_duration_sec + self.source_seek_duration_bonus_sec,
            self.max_source_seek_extended_duration_sec,
        )

    def _reward_source_seek_duration(self):
        if self.duration_reward_per_local_goal_sec <= 0.0:
            return
        old_limit = self._effective_source_seek_duration_sec()
        self.source_seek_successful_local_goals += 1
        self.source_seek_duration_bonus_sec = min(
            self.source_seek_duration_bonus_sec + self.duration_reward_per_local_goal_sec,
            max(0.0, self.max_source_seek_extended_duration_sec - self.max_source_seek_duration_sec),
        )
        new_limit = self._effective_source_seek_duration_sec()
        if new_limit > old_limit:
            self.get_logger().info(
                "Rewarded source seek duration after local goal success: "
                f"{old_limit:.1f}s -> {new_limit:.1f}s "
                f"(success_count={self.source_seek_successful_local_goals})"
            )

    def _source_reached(self) -> bool:
        if self.robot_pose is None or self.source_estimate_xy is None:
            return False
        if self._state_age_sec() < self.source_success_min_duration_sec:
            return False
        if self.pmax < self.pmax_success_threshold:
            return False
        distance = math.hypot(
            self.robot_pose[0] - self.source_estimate_xy[0],
            self.robot_pose[1] - self.source_estimate_xy[1],
        )
        return distance <= self.source_reached_distance

    def _entropy_not_improving(self) -> bool:
        if self.entropy is None or self.source_seek_start_entropy is None:
            return False
        if self._state_age_sec() < self.entropy_grace_sec:
            return False
        return self.entropy > self.source_seek_start_entropy - self.min_entropy_drop

    def _enter_state(self, new_state: ControlState, reason: str):
        if self.state == new_state:
            return
        old_state = self.state
        self.state = new_state
        self.state_enter_time = self.get_clock().now()

        if new_state == ControlState.SOURCE_SEEK:
            self.source_seek_peak_pmax = self.pmax
            self.source_seek_start_entropy = self.entropy
            self.source_seek_start_xy = None
            self.source_seek_duration_bonus_sec = 0.0
            self.source_seek_successful_local_goals = 0
            self.local_goal_result_status = None
            self.no_valid_goal_attempts = 0
            if self.robot_pose is not None:
                self.source_seek_start_xy = (self.robot_pose[0], self.robot_pose[1])
        elif new_state == ControlState.PATROL:
            self._cancel_local_goal()
            self.source_seek_start_xy = None
            self.source_seek_start_entropy = None
            if old_state == ControlState.RECOVERY:
                self.patrol_resume_until = self.get_clock().now() + Duration(
                    seconds=self.patrol_resume_lockout_sec
                )
        elif new_state == ControlState.RECOVERY:
            if reason == "source_reached":
                self.source_reached_latched = True
            if reason.startswith("no_valid_local_nav2_goal") or reason.startswith("local_goal_"):
                self.source_seek_cooldown_until = self.get_clock().now() + Duration(
                    seconds=self.source_seek_failure_cooldown_sec
                )
            self._cancel_local_goal()

        self.get_logger().info(f"{old_state.value} -> {new_state.value}: {reason}")
        self._publish_status(f"{old_state.value}->{new_state.value} reason={reason}")

    def _patrol_resume_locked(self) -> bool:
        if self.patrol_resume_until is None:
            return False
        if self.get_clock().now() >= self.patrol_resume_until:
            self.patrol_resume_until = None
            return False
        return True

    def _source_seek_cooldown_locked(self) -> bool:
        if self.source_seek_cooldown_until is None:
            return False
        if self.get_clock().now() >= self.source_seek_cooldown_until:
            self.source_seek_cooldown_until = None
            return False
        return True

    def _state_age_sec(self) -> float:
        return self._age_sec(self.state_enter_time)

    def _age_sec(self, stamp) -> float:
        return (self.get_clock().now() - stamp).nanoseconds * 1e-9

    def _world_to_grid(self, grid: OccupancyGrid, x: float, y: float) -> Optional[Tuple[int, int]]:
        col = int(math.floor((x - grid.info.origin.position.x) / grid.info.resolution))
        row = int(math.floor((y - grid.info.origin.position.y) / grid.info.resolution))
        if not self._grid_contains(grid, col, row):
            return None
        return col, row

    def _grid_to_world(self, grid: OccupancyGrid, col: int, row: int) -> Tuple[float, float]:
        x = grid.info.origin.position.x + (col + 0.5) * grid.info.resolution
        y = grid.info.origin.position.y + (row + 0.5) * grid.info.resolution
        return x, y

    def _grid_contains(self, grid: OccupancyGrid, col: int, row: int) -> bool:
        return 0 <= col < grid.info.width and 0 <= row < grid.info.height

    def _publish_status(self, text: str):
        self.status_pub.publish(String(data=text))


def main(args=None):
    rclpy.init(args=args)
    node = VInfoArbiter()
    try:
        rclpy.spin(node)
    finally:
        node.cmd_pub.publish(Twist())
        node._cancel_local_goal()
        node.active_pub.publish(Bool(data=False))
        node.success_pub.publish(Bool(data=node.source_reached_latched))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
