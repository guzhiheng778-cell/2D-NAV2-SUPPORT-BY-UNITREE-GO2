#!/usr/bin/env python3

import csv
import math
import os
from dataclasses import dataclass
from typing import List

import rclpy
import yaml
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry, Path
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node


@dataclass
class PatrolPoint:
    name: str
    x: float
    y: float
    yaw: float


def yaw_to_quaternion(yaw: float):
    half_yaw = yaw * 0.5
    return 0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)


def quaternion_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class PatrolManager(Node):
    def __init__(self):
        super().__init__("patrol_manager")

        self.declare_parameter("points_file", "")
        self.declare_parameter("loop", False)
        self.declare_parameter("wait_time_sec", 3.0)
        self.declare_parameter("goal_settle_time_sec", 2.0)
        self.declare_parameter("goal_verify_tolerance", 0.30)
        self.declare_parameter("goal_verify_yaw_tolerance", 0.60)
        self.declare_parameter("max_goal_retries", 2)
        self.declare_parameter("goal_timeout_sec", 120.0)
        self.declare_parameter("goal_frame", "map")#如果你的 Nav2 使用 AMCL 或 SLAM 定位，目标点一般都应该在 map 坐标系下
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("robot_pose_topic", "/amcl_pose")
        self.declare_parameter("trajectory_topic", "/patrol/trajectory")
        self.declare_parameter("trajectory_output_file", "/tmp/go2_patrol_trajectory.csv")

        self.loop = self.get_parameter("loop").value
        self.wait_time_sec = float(self.get_parameter("wait_time_sec").value)
        self.goal_settle_time_sec = float(self.get_parameter("goal_settle_time_sec").value)
        self.goal_verify_tolerance = float(self.get_parameter("goal_verify_tolerance").value)
        self.goal_verify_yaw_tolerance = float(self.get_parameter("goal_verify_yaw_tolerance").value)
        self.max_goal_retries = int(self.get_parameter("max_goal_retries").value)
        self.goal_timeout_sec = float(self.get_parameter("goal_timeout_sec").value)
        self.goal_frame = self.get_parameter("goal_frame").value
        odom_topic = self.get_parameter("odom_topic").value
        robot_pose_topic = self.get_parameter("robot_pose_topic").value
        trajectory_topic = self.get_parameter("trajectory_topic").value
        self.trajectory_output_file = self.get_parameter("trajectory_output_file").value

        self.points = self._load_patrol_points()
        self.current_index = 0
        self.active_goal_name = ""
        self.goal_in_progress = False
        self.wait_until = None
        self.goal_sent_time = None
        self.has_map_pose = False
        self.latest_pose_frame = ""
        self.latest_pose_x = None
        self.latest_pose_y = None
        self.latest_pose_yaw = None
        self.active_goal_retry_count = 0

        self.path_msg = Path()
        self.path_msg.header.frame_id = self.goal_frame
        self.trajectory_pub = self.create_publisher(Path, trajectory_topic, 10)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self._odom_callback, 20)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, robot_pose_topic, self._pose_callback, 10
        )#只要这个话题收到新消息，就自动调用 self._pose_callback(msg)。
        self.action_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        self._init_trajectory_file()
        self.timer = self.create_timer(0.5, self._tick)

        names = ", ".join(point.name for point in self.points)
        self.get_logger().info(f"Patrol manager ready with {len(self.points)} points: {names}")

    def _load_patrol_points(self) -> List[PatrolPoint]:
        points_file = self.get_parameter("points_file").value
        if not points_file:
            raise ValueError("points_file parameter is empty")
        with open(points_file, "r") as yaml_file:
            data = yaml.safe_load(yaml_file) or {}

        raw_points = data.get("patrol_points", [])
        points = []
        for index, raw_point in enumerate(raw_points):
            if not isinstance(raw_point, dict):
                raise ValueError("Each patrol point must be a YAML dictionary")
            points.append(
                PatrolPoint(
                    name=str(raw_point.get("name", f"point_{index}")),
                    x=float(raw_point["x"]),
                    y=float(raw_point["y"]),
                    yaw=float(raw_point.get("yaw", 0.0)),
                )
            )

        if not points:
            raise ValueError("patrol_points is empty")
        return points

    def _init_trajectory_file(self):
        directory = os.path.dirname(self.trajectory_output_file)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.trajectory_output_file, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["stamp_sec", "frame_id", "x", "y", "yaw", "active_goal"])

    def _odom_callback(self, msg: Odometry):
        if self.has_map_pose:
            return
        self._record_pose(msg.header, msg.pose.pose)

    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        self.has_map_pose = True
        self._record_pose(msg.header, msg.pose.pose)

    def _record_pose(self, header, pose_msg):
        pose = PoseStamped()
        pose.header = header
        pose.pose = pose_msg
        self.path_msg.header.frame_id = header.frame_id
        self.path_msg.header.stamp = self.get_clock().now().to_msg()
        self.path_msg.poses.append(pose)
        self.trajectory_pub.publish(self.path_msg)

        yaw = quaternion_to_yaw(pose_msg.orientation)
        self.latest_pose_frame = header.frame_id
        self.latest_pose_x = pose_msg.position.x
        self.latest_pose_y = pose_msg.position.y
        self.latest_pose_yaw = yaw
        stamp_sec = header.stamp.sec + header.stamp.nanosec * 1e-9
        with open(self.trajectory_output_file, "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                f"{stamp_sec:.9f}",
                header.frame_id,
                f"{pose_msg.position.x:.4f}",
                f"{pose_msg.position.y:.4f}",
                f"{yaw:.4f}",
                self.active_goal_name,
            ])

    def _tick(self):
        if self.goal_in_progress:
            if self.goal_sent_time is None:
                return
            elapsed = self.get_clock().now() - self.goal_sent_time
            if elapsed > Duration(seconds=self.goal_timeout_sec):
                self.get_logger().warn(f"Goal '{self.active_goal_name}' timed out")
                self.goal_in_progress = False
                self._advance_point()
            return

        if self.wait_until is not None:
            if self.get_clock().now() < self.wait_until:
                return
            self.wait_until = None
            self._verify_reached_then_advance()
            return

        if self.current_index >= len(self.points):
            if not self.loop:
                self.get_logger().info("Patrol finished")
                self.timer.cancel()
                return
            self.current_index = 0

        self._send_current_goal()

    def _send_current_goal(self):
        if not self.action_client.wait_for_server(timeout_sec=0.1):
            self.get_logger().info("Waiting for Nav2 navigate_to_pose action server...")
            return

        point = self.points[self.current_index]
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = self.goal_frame
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = point.x
        goal_msg.pose.pose.position.y = point.y
        qx, qy, qz, qw = yaw_to_quaternion(point.yaw)
        goal_msg.pose.pose.orientation.x = qx
        goal_msg.pose.pose.orientation.y = qy
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        self.active_goal_name = point.name
        self.goal_in_progress = True
        self.goal_sent_time = self.get_clock().now()
        self.get_logger().info(
            f"Sending patrol goal {self.current_index + 1}/{len(self.points)} "
            f"'{point.name}' at x={point.x:.2f}, y={point.y:.2f}, yaw={point.yaw:.2f}"
        )

        send_future = self.action_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(f"Goal '{self.active_goal_name}' was rejected")
            self.goal_in_progress = False
            self._advance_point()
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._goal_result_callback)

    def _goal_result_callback(self, future):
        status = future.result().status
        if status == 4:
            self.get_logger().info(
                f"Reached '{self.active_goal_name}', waiting "
                f"{self.wait_time_sec + self.goal_settle_time_sec:.1f}s before verification"
            )
            self.wait_until = self.get_clock().now() + Duration(
                seconds=self.wait_time_sec + self.goal_settle_time_sec
            )
        else:
            self.get_logger().warn(f"Goal '{self.active_goal_name}' finished with status {status}")
            self._advance_point()
        self.goal_in_progress = False

    def _verify_reached_then_advance(self):
        point = self.points[self.current_index]
        if (
            self.latest_pose_x is None
            or self.latest_pose_y is None
            or self.latest_pose_yaw is None
            or self.latest_pose_frame != self.goal_frame
        ):
            self.get_logger().warn(
                f"Cannot verify '{point.name}' in frame '{self.goal_frame}', advancing anyway"
            )
            self._advance_point()
            return

        dx = self.latest_pose_x - point.x
        dy = self.latest_pose_y - point.y
        distance = math.hypot(dx, dy)
        yaw_error = math.atan2(
            math.sin(self.latest_pose_yaw - point.yaw),
            math.cos(self.latest_pose_yaw - point.yaw),
        )
        yaw_error = abs(yaw_error)

        if distance <= self.goal_verify_tolerance and yaw_error <= self.goal_verify_yaw_tolerance:
            self.get_logger().info(
                f"Verified '{point.name}': distance={distance:.2f}m, yaw_error={yaw_error:.2f}rad"
            )
            self._advance_point()
            return

        if self.active_goal_retry_count >= self.max_goal_retries:
            self.get_logger().warn(
                f"Verification failed for '{point.name}' after {self.active_goal_retry_count} retries: "
                f"distance={distance:.2f}m, yaw_error={yaw_error:.2f}rad; advancing"
            )
            self._advance_point()
            return

        self.active_goal_retry_count += 1
        self.get_logger().warn(
            f"Verification failed for '{point.name}': distance={distance:.2f}m, "
            f"yaw_error={yaw_error:.2f}rad; retry "
            f"{self.active_goal_retry_count}/{self.max_goal_retries}"
        )
        self._send_current_goal()

    def _advance_point(self):
        self.current_index += 1
        self.active_goal_name = ""
        self.goal_sent_time = None
        self.active_goal_retry_count = 0


def main(args=None):
    rclpy.init(args=args)
    node = PatrolManager()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
