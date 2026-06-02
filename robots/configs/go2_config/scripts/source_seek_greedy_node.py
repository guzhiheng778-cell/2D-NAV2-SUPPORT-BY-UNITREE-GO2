#!/usr/bin/env python3

import math
from typing import Optional, Tuple

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


def quaternion_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def yaw_to_quaternion(yaw: float):
    half_yaw = yaw * 0.5
    return 0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)


class SourceSeekGreedy(Node):
    def __init__(self):
        super().__init__("source_seek_greedy")

        self.declare_parameter("pmax_topic", "/pmax")
        self.declare_parameter("source_estimate_topic", "/source_estimate")
        self.declare_parameter("robot_pose_topic", "/amcl_pose")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("status_topic", "/source_seek/status")
        self.declare_parameter("active_topic", "/source_seek/active")
        self.declare_parameter("goal_frame", "map")
        self.declare_parameter("control_rate_hz", 10.0)
        self.declare_parameter("pmax_activate_threshold", 0.004)
        self.declare_parameter("pmax_deactivate_threshold", 0.002)
        self.declare_parameter("goal_tolerance", 0.35)
        self.declare_parameter("min_goal_update_distance", 0.50)
        self.declare_parameter("min_goal_update_period_sec", 5.0)
        self.declare_parameter("goal_timeout_sec", 120.0)

        self.pmax_activate_threshold = float(self.get_parameter("pmax_activate_threshold").value)
        self.pmax_deactivate_threshold = float(self.get_parameter("pmax_deactivate_threshold").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.goal_frame = str(self.get_parameter("goal_frame").value)
        self.min_goal_update_distance = float(self.get_parameter("min_goal_update_distance").value)
        self.min_goal_update_period = Duration(
            seconds=float(self.get_parameter("min_goal_update_period_sec").value)
        )
        self.goal_timeout = Duration(seconds=float(self.get_parameter("goal_timeout_sec").value))

        self.robot_pose: Optional[Tuple[float, float, float]] = None
        self.source_xy: Optional[Tuple[float, float]] = None
        self.pmax = 0.0
        self.active = False
        self.arrived = False
        self.has_map_pose = False
        self.goal_in_progress = False
        self.current_goal_handle = None
        self.current_goal_xy: Optional[Tuple[float, float]] = None
        self.last_goal_sent_time = None

        self.status_pub = self.create_publisher(String, self.get_parameter("status_topic").value, 10)
        self.active_pub = self.create_publisher(Bool, self.get_parameter("active_topic").value, 10)
        self.action_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        self.create_subscription(
            Float32, self.get_parameter("pmax_topic").value, self._pmax_callback, 10
        )
        self.create_subscription(
            PoseStamped,
            self.get_parameter("source_estimate_topic").value,
            self._source_callback,
            10,
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

        control_rate_hz = max(1.0, float(self.get_parameter("control_rate_hz").value))
        self.timer = self.create_timer(1.0 / control_rate_hz, self._control_tick)
        self.get_logger().info(
            "Greedy source seeking ready: "
            f"activate Pmax>={self.pmax_activate_threshold:.4f}, "
            f"deactivate Pmax<{self.pmax_deactivate_threshold:.4f}"
        )

    def _pmax_callback(self, msg: Float32):
        self.pmax = float(msg.data)

    def _source_callback(self, msg: PoseStamped):
        self.source_xy = (msg.pose.position.x, msg.pose.position.y)

    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        self.has_map_pose = True
        self.robot_pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            quaternion_to_yaw(msg.pose.pose.orientation),
        )

    def _odom_callback(self, msg: Odometry):
        if self.has_map_pose:
            return
        self.robot_pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            quaternion_to_yaw(msg.pose.pose.orientation),
        )

    def _control_tick(self):
        self._update_active_state()
        self.active_pub.publish(Bool(data=self.active))

        if not self.active:
            self._publish_status("patrol_or_idle")
            return

        if self.robot_pose is None or self.source_xy is None:
            self._publish_status("waiting_for_pose_or_source")
            return

        distance = self._distance_to_source()
        if distance <= self.goal_tolerance:
            self.arrived = True
            self._publish_status(
                f"arrived distance={distance:.2f} pmax={self.pmax:.4f}"
            )
            return

        self.arrived = False
        if self.goal_in_progress:
            if self._goal_timed_out():
                self.get_logger().warn("Source seek Nav2 goal timed out, canceling")
                self._cancel_current_goal()
            elif self._source_moved_enough() and self._goal_update_period_elapsed():
                self.get_logger().info("Source estimate moved, updating Nav2 goal")
                self._cancel_current_goal()
                self._send_source_goal()
            else:
                self._publish_status(
                    f"nav2_seeking distance={distance:.2f} pmax={self.pmax:.4f}"
                )
            return

        self._send_source_goal()

    def _update_active_state(self):
        if self.active:
            if self.pmax < self.pmax_deactivate_threshold:
                self.active = False
                self.arrived = False
                self._cancel_current_goal()
            return

        if self.pmax >= self.pmax_activate_threshold:
            self.active = True
            self.arrived = False

    def _distance_to_source(self) -> float:
        robot_x, robot_y, _ = self.robot_pose
        source_x, source_y = self.source_xy
        return math.hypot(source_x - robot_x, source_y - robot_y)

    def _source_moved_enough(self) -> bool:
        if self.current_goal_xy is None or self.source_xy is None:
            return True
        dx = self.source_xy[0] - self.current_goal_xy[0]
        dy = self.source_xy[1] - self.current_goal_xy[1]
        return math.hypot(dx, dy) >= self.min_goal_update_distance

    def _goal_update_period_elapsed(self) -> bool:
        if self.last_goal_sent_time is None:
            return True
        return self.get_clock().now() - self.last_goal_sent_time >= self.min_goal_update_period

    def _goal_timed_out(self) -> bool:
        if self.last_goal_sent_time is None:
            return False
        return self.get_clock().now() - self.last_goal_sent_time >= self.goal_timeout

    def _send_source_goal(self):
        if self.source_xy is None:
            return
        if not self.action_client.wait_for_server(timeout_sec=0.1):
            self._publish_status("waiting_for_nav2_action_server")
            return

        source_x, source_y = self.source_xy
        yaw = 0.0
        if self.robot_pose is not None:
            robot_x, robot_y, _ = self.robot_pose
            yaw = math.atan2(source_y - robot_y, source_x - robot_x)

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = self.goal_frame
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = source_x
        goal_msg.pose.pose.position.y = source_y
        qx, qy, qz, qw = yaw_to_quaternion(yaw)
        goal_msg.pose.pose.orientation.x = qx
        goal_msg.pose.pose.orientation.y = qy
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        self.goal_in_progress = True
        self.current_goal_xy = (source_x, source_y)
        self.last_goal_sent_time = self.get_clock().now()
        self._publish_status(
            f"sending_nav2_goal x={source_x:.2f} y={source_y:.2f} pmax={self.pmax:.4f}"
        )
        send_future = self.action_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Source seek Nav2 goal was rejected")
            self.goal_in_progress = False
            self.current_goal_handle = None
            return

        self.current_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._goal_result_callback)

    def _goal_result_callback(self, future):
        result = future.result()
        status = result.status
        self.goal_in_progress = False
        self.current_goal_handle = None

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.arrived = True
            self._publish_status(f"nav2_arrived pmax={self.pmax:.4f}")
            return

        if self.active:
            self._publish_status(f"nav2_goal_finished status={status}, will retry")

    def _cancel_current_goal(self):
        if self.current_goal_handle is not None:
            self.current_goal_handle.cancel_goal_async()
        self.goal_in_progress = False
        self.current_goal_handle = None

    def _publish_status(self, text: str):
        self.status_pub.publish(String(data=text))


def main(args=None):
    rclpy.init(args=args)
    node = SourceSeekGreedy()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
