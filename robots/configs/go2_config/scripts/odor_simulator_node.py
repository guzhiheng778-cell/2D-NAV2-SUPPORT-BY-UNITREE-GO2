#!/usr/bin/env python3

import math
import random
from collections import deque

import rclpy
from geometry_msgs.msg import Point, PoseWithCovarianceStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import Bool, ColorRGBA, Float32
from visualization_msgs.msg import Marker, MarkerArray


class OdorSimulator(Node):
    def __init__(self):
        super().__init__("odor_simulator")

        self.declare_parameter("frame_id", "map")
        self.declare_parameter("publish_rate_hz", 5.0)
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("robot_pose_topic", "/amcl_pose")
        self.declare_parameter("wind_topic", "/wind_vector")
        self.declare_parameter("concentration_topic", "/odor_concentration")
        self.declare_parameter("detection_topic", "/odor_detection")
        self.declare_parameter("marker_topic", "/odor/markers")
        self.declare_parameter("source_x", -2.0)
        self.declare_parameter("source_y", 0.5)
        self.declare_parameter("source_z", 0.2)
        self.declare_parameter("release_rate", 8.0)
        self.declare_parameter("plume_sigma_y_base", 0.20)
        self.declare_parameter("plume_sigma_y_gain", 0.22)
        self.declare_parameter("plume_sigma_z_base", 0.20)
        self.declare_parameter("plume_sigma_z_gain", 0.12)
        self.declare_parameter("sensor_height", 0.25)
        self.declare_parameter("detection_threshold", 0.18)
        self.declare_parameter("turbulence_noise_std", 0.04)
        self.declare_parameter("hit_marker_lifetime_sec", 120.0)
        self.declare_parameter("max_hit_markers", 300)

        self.frame_id = self.get_parameter("frame_id").value
        self.source_x = float(self.get_parameter("source_x").value)
        self.source_y = float(self.get_parameter("source_y").value)
        self.source_z = float(self.get_parameter("source_z").value)
        self.release_rate = float(self.get_parameter("release_rate").value)
        self.sigma_y_base = float(self.get_parameter("plume_sigma_y_base").value)
        self.sigma_y_gain = float(self.get_parameter("plume_sigma_y_gain").value)
        self.sigma_z_base = float(self.get_parameter("plume_sigma_z_base").value)
        self.sigma_z_gain = float(self.get_parameter("plume_sigma_z_gain").value)
        self.sensor_height = float(self.get_parameter("sensor_height").value)
        self.detection_threshold = float(self.get_parameter("detection_threshold").value)
        self.turbulence_noise_std = float(self.get_parameter("turbulence_noise_std").value)
        self.hit_marker_lifetime_sec = float(self.get_parameter("hit_marker_lifetime_sec").value)
        max_hit_markers = int(self.get_parameter("max_hit_markers").value)

        odom_topic = self.get_parameter("odom_topic").value
        robot_pose_topic = self.get_parameter("robot_pose_topic").value
        wind_topic = self.get_parameter("wind_topic").value
        concentration_topic = self.get_parameter("concentration_topic").value
        detection_topic = self.get_parameter("detection_topic").value
        marker_topic = self.get_parameter("marker_topic").value
        publish_rate_hz = max(0.1, float(self.get_parameter("publish_rate_hz").value))

        self.robot_x = None
        self.robot_y = None
        self.has_map_pose = False
        self.wind_x = 1.0
        self.wind_y = 0.0
        self.hit_points = deque(maxlen=max_hit_markers)
        self.last_detection = False

        self.concentration_pub = self.create_publisher(Float32, concentration_topic, 10)
        self.detection_pub = self.create_publisher(Bool, detection_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, marker_topic, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_callback, 20)
        self.create_subscription(PoseWithCovarianceStamped, robot_pose_topic, self._pose_callback, 10)
        self.create_subscription(Vector3Stamped, wind_topic, self._wind_callback, 10)
        self.timer = self.create_timer(1.0 / publish_rate_hz, self._tick)

        self.get_logger().info(
            f"Odor simulator source=({self.source_x:.2f}, {self.source_y:.2f}), "
            f"threshold={self.detection_threshold:.2f}"
        )

    def _odom_callback(self, msg: Odometry):
        if self.has_map_pose:
            return
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        self.has_map_pose = True
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

    def _wind_callback(self, msg: Vector3Stamped):
        self.wind_x = msg.vector.x
        self.wind_y = msg.vector.y

    def _tick(self):
        self._publish_markers()
        if self.robot_x is None or self.robot_y is None:
            return

        concentration = self._compute_concentration(self.robot_x, self.robot_y)
        detected = concentration >= self.detection_threshold

        self.concentration_pub.publish(Float32(data=float(concentration)))
        self.detection_pub.publish(Bool(data=detected))

        if detected:
            self.hit_points.append((self.robot_x, self.robot_y, concentration))
        if detected != self.last_detection:
            state = "hit" if detected else "void"
            self.get_logger().info(f"Odor {state}: concentration={concentration:.3f}")
        self.last_detection = detected

    def _compute_concentration(self, robot_x: float, robot_y: float) -> float:
        wind_speed = max(0.05, math.hypot(self.wind_x, self.wind_y))
        wind_yaw = math.atan2(self.wind_y, self.wind_x)

        dx = robot_x - self.source_x
        dy = robot_y - self.source_y
        downwind = dx * math.cos(wind_yaw) + dy * math.sin(wind_yaw)
        crosswind = -dx * math.sin(wind_yaw) + dy * math.cos(wind_yaw)

        if downwind <= 0.0:
            return max(0.0, random.gauss(0.0, self.turbulence_noise_std * 0.25))

        sigma_y = self.sigma_y_base + self.sigma_y_gain * downwind
        sigma_z = self.sigma_z_base + self.sigma_z_gain * downwind
        vertical = self.sensor_height - self.source_z
        plume = self.release_rate / (2.0 * math.pi * wind_speed * sigma_y * sigma_z)
        plume *= math.exp(-0.5 * (crosswind / sigma_y) ** 2)
        plume *= math.exp(-0.5 * (vertical / sigma_z) ** 2)

        noisy = plume + random.gauss(0.0, self.turbulence_noise_std * max(1.0, plume))
        return max(0.0, noisy)

    def _publish_markers(self):
        now = self.get_clock().now().to_msg()
        markers = MarkerArray()
        markers.markers.append(self._source_marker(now))
        markers.markers.append(self._wind_marker(now))
        markers.markers.append(self._hit_points_marker(now))
        self.marker_pub.publish(markers)

    def _source_marker(self, stamp):
        marker = self._base_marker(stamp, 0, Marker.SPHERE)
        marker.pose.position.x = self.source_x
        marker.pose.position.y = self.source_y
        marker.pose.position.z = self.source_z
        marker.scale.x = 0.35
        marker.scale.y = 0.35
        marker.scale.z = 0.35
        marker.color = ColorRGBA(r=1.0, g=0.35, b=0.0, a=0.9)
        return marker

    def _wind_marker(self, stamp):
        marker = self._base_marker(stamp, 1, Marker.ARROW)
        wind_speed = max(0.05, math.hypot(self.wind_x, self.wind_y))
        wind_yaw = math.atan2(self.wind_y, self.wind_x)
        length = min(2.5, 0.8 + wind_speed)
        start = Point(x=self.source_x, y=self.source_y, z=0.7)
        end = Point(
            x=self.source_x + length * math.cos(wind_yaw),
            y=self.source_y + length * math.sin(wind_yaw),
            z=0.7,
        )
        marker.points = [start, end]
        marker.scale.x = 0.06
        marker.scale.y = 0.18
        marker.scale.z = 0.18
        marker.color = ColorRGBA(r=0.1, g=0.55, b=1.0, a=0.9)
        return marker

    def _hit_points_marker(self, stamp):
        marker = self._base_marker(stamp, 2, Marker.POINTS)
        marker.scale.x = 0.12
        marker.scale.y = 0.12
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.1, a=0.95)
        marker.lifetime = Duration(seconds=self.hit_marker_lifetime_sec).to_msg()
        marker.points = [
            Point(x=x, y=y, z=0.08)
            for x, y, _concentration in self.hit_points
        ]
        return marker

    def _base_marker(self, stamp, marker_id: int, marker_type: int):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.frame_id
        marker.ns = "odor_simulation"
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker


def main(args=None):
    rclpy.init(args=args)
    node = OdorSimulator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
