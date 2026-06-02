#!/usr/bin/env python3

import math
import random

import rclpy
from geometry_msgs.msg import Vector3Stamped
from rclpy.node import Node


class WindSimulator(Node):
    def __init__(self):
        super().__init__("wind_simulator")

        self.declare_parameter("frame_id", "map")
        self.declare_parameter("publish_rate_hz", 2.0)
        self.declare_parameter("speed_mps", 1.2)
        self.declare_parameter("direction_yaw", 0.0)
        self.declare_parameter("speed_noise_std", 0.08)
        self.declare_parameter("direction_noise_std", 0.08)

        self.frame_id = self.get_parameter("frame_id").value
        self.speed_mps = float(self.get_parameter("speed_mps").value)
        self.direction_yaw = float(self.get_parameter("direction_yaw").value)
        self.speed_noise_std = float(self.get_parameter("speed_noise_std").value)
        self.direction_noise_std = float(self.get_parameter("direction_noise_std").value)

        publish_rate_hz = max(0.1, float(self.get_parameter("publish_rate_hz").value))
        self.publisher = self.create_publisher(Vector3Stamped, "/wind_vector", 10)
        self.timer = self.create_timer(1.0 / publish_rate_hz, self._publish_wind)
        self.get_logger().info(
            f"Wind simulator publishing /wind_vector at {publish_rate_hz:.1f} Hz"
        )

    def _publish_wind(self):
        speed = max(0.0, self.speed_mps + random.gauss(0.0, self.speed_noise_std))
        yaw = self.direction_yaw + random.gauss(0.0, self.direction_noise_std)

        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.vector.x = speed * math.cos(yaw)
        msg.vector.y = speed * math.sin(yaw)
        msg.vector.z = 0.0
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = WindSimulator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
