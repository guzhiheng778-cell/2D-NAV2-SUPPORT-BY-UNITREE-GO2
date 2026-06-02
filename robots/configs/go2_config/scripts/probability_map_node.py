#!/usr/bin/env python3

import math
import csv
import os
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import Point, PoseStamped, Vector3Stamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, ColorRGBA, Float32
from visualization_msgs.msg import Marker, MarkerArray


class ProbabilityMap(Node):
    def __init__(self):
        super().__init__("probability_map")

        self.declare_parameter("frame_id", "map")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("wind_topic", "/wind_vector")
        self.declare_parameter("detection_topic", "/odor_detection")
        self.declare_parameter("probability_grid_topic", "/probability_grid")
        self.declare_parameter("entropy_topic", "/entropy")
        self.declare_parameter("pmax_topic", "/pmax")
        self.declare_parameter("source_estimate_topic", "/source_estimate")
        self.declare_parameter("marker_topic", "/probability/markers")
        self.declare_parameter("csv_output_file", "/tmp/go2_probability_metrics.csv")
        self.declare_parameter("map_min_x", -10.0)
        self.declare_parameter("map_max_x", 10.0)
        self.declare_parameter("map_min_y", -10.0)
        self.declare_parameter("map_max_y", 10.0)
        self.declare_parameter("resolution", 0.20)
        self.declare_parameter("publish_rate_hz", 2.0)
        self.declare_parameter("min_update_distance", 0.15)
        self.declare_parameter("min_update_interval_sec", 0.50)
        self.declare_parameter("release_rate", 8.0)
        self.declare_parameter("plume_sigma_y_base", 0.20)
        self.declare_parameter("plume_sigma_y_gain", 0.22)
        self.declare_parameter("plume_sigma_z_base", 0.20)
        self.declare_parameter("plume_sigma_z_gain", 0.12)
        self.declare_parameter("sensor_height", 0.25)
        self.declare_parameter("source_height", 0.2)
        self.declare_parameter("detection_threshold", 0.18)
        self.declare_parameter("hit_likelihood_floor", 0.05)
        self.declare_parameter("void_likelihood_floor", 0.05)
        self.declare_parameter("likelihood_temperature", 0.12)

        self.frame_id = self.get_parameter("frame_id").value
        self.map_min_x = float(self.get_parameter("map_min_x").value)
        self.map_max_x = float(self.get_parameter("map_max_x").value)
        self.map_min_y = float(self.get_parameter("map_min_y").value)
        self.map_max_y = float(self.get_parameter("map_max_y").value)
        self.resolution = float(self.get_parameter("resolution").value)
        self.min_update_distance = float(self.get_parameter("min_update_distance").value)
        self.min_update_interval_sec = float(self.get_parameter("min_update_interval_sec").value)
        self.release_rate = float(self.get_parameter("release_rate").value)
        self.sigma_y_base = float(self.get_parameter("plume_sigma_y_base").value)
        self.sigma_y_gain = float(self.get_parameter("plume_sigma_y_gain").value)
        self.sigma_z_base = float(self.get_parameter("plume_sigma_z_base").value)
        self.sigma_z_gain = float(self.get_parameter("plume_sigma_z_gain").value)
        self.sensor_height = float(self.get_parameter("sensor_height").value)
        self.source_height = float(self.get_parameter("source_height").value)
        self.detection_threshold = float(self.get_parameter("detection_threshold").value)
        self.hit_likelihood_floor = float(self.get_parameter("hit_likelihood_floor").value)
        self.void_likelihood_floor = float(self.get_parameter("void_likelihood_floor").value)
        self.likelihood_temperature = max(
            1e-3, float(self.get_parameter("likelihood_temperature").value)
        )
        self.csv_output_file = self.get_parameter("csv_output_file").value

        self.width = int(math.ceil((self.map_max_x - self.map_min_x) / self.resolution))
        self.height = int(math.ceil((self.map_max_y - self.map_min_y) / self.resolution))
        self.cell_count = self.width * self.height
        self.probabilities = [1.0 / self.cell_count] * self.cell_count

        self.robot_xy: Optional[Tuple[float, float]] = None
        self.wind_xy = (1.0, 0.0)
        self.last_update_xy: Optional[Tuple[float, float]] = None
        self.last_update_time = None
        self.update_count = 0
        self.latest_detection = None

        self.grid_pub = self.create_publisher(
            OccupancyGrid, self.get_parameter("probability_grid_topic").value, 1
        )
        self.entropy_pub = self.create_publisher(
            Float32, self.get_parameter("entropy_topic").value, 10
        )
        self.pmax_pub = self.create_publisher(
            Float32, self.get_parameter("pmax_topic").value, 10
        )
        self.source_pub = self.create_publisher(
            PoseStamped, self.get_parameter("source_estimate_topic").value, 10
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, self.get_parameter("marker_topic").value, 10
        )

        self.create_subscription(
            Odometry, self.get_parameter("odom_topic").value, self._odom_callback, 20
        )
        self.create_subscription(
            Vector3Stamped, self.get_parameter("wind_topic").value, self._wind_callback, 10
        )
        self.create_subscription(
            Bool, self.get_parameter("detection_topic").value, self._detection_callback, 20
        )

        publish_rate_hz = max(0.1, float(self.get_parameter("publish_rate_hz").value))
        self.timer = self.create_timer(1.0 / publish_rate_hz, self._publish_outputs)
        self._init_csv()
        self.get_logger().info(
            f"Probability map ready: {self.width}x{self.height}, resolution={self.resolution:.2f} m"
        )

    def _odom_callback(self, msg: Odometry):
        self.robot_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def _wind_callback(self, msg: Vector3Stamped):
        self.wind_xy = (msg.vector.x, msg.vector.y)

    def _detection_callback(self, msg: Bool):
        if self.robot_xy is None:
            return
        if not self._should_update():
            return
        self._bayes_update(msg.data)
        self.last_update_xy = self.robot_xy
        self.last_update_time = self.get_clock().now()
        self.latest_detection = msg.data
        self.update_count += 1
        self._record_metrics(msg.data)

    def _should_update(self) -> bool:
        now = self.get_clock().now()
        if self.last_update_time is not None:
            elapsed = (now - self.last_update_time).nanoseconds * 1e-9
            if elapsed < self.min_update_interval_sec:
                return False

        if self.last_update_xy is None:
            return True
        dx = self.robot_xy[0] - self.last_update_xy[0]
        dy = self.robot_xy[1] - self.last_update_xy[1]
        return math.hypot(dx, dy) >= self.min_update_distance

    def _bayes_update(self, detected: bool):
        updated = []
        for index, prior in enumerate(self.probabilities):
            source_x, source_y = self._cell_center(index)
            p_hit = self._hit_probability(source_x, source_y)
            likelihood = p_hit if detected else (1.0 - p_hit)
            if detected:
                likelihood = max(self.hit_likelihood_floor, likelihood)
            else:
                likelihood = max(self.void_likelihood_floor, likelihood)
            updated.append(prior * likelihood)

        total = sum(updated)
        if total <= 0.0 or not math.isfinite(total):
            self.get_logger().warn("Probability update underflow, resetting to uniform prior")
            self.probabilities = [1.0 / self.cell_count] * self.cell_count
            return
        self.probabilities = [value / total for value in updated]

    def _hit_probability(self, source_x: float, source_y: float) -> float:
        concentration = self._expected_concentration(source_x, source_y)
        return 1.0 / (1.0 + math.exp(-(concentration - self.detection_threshold) / self.likelihood_temperature))

    def _expected_concentration(self, source_x: float, source_y: float) -> float:
        robot_x, robot_y = self.robot_xy
        wind_x, wind_y = self.wind_xy
        wind_speed = max(0.05, math.hypot(wind_x, wind_y))
        wind_yaw = math.atan2(wind_y, wind_x)

        dx = robot_x - source_x
        dy = robot_y - source_y
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

    def _publish_outputs(self):
        now = self.get_clock().now().to_msg()
        entropy = self._entropy()
        pmax, source_x, source_y = self._source_estimate()

        self.entropy_pub.publish(Float32(data=float(entropy)))
        self.pmax_pub.publish(Float32(data=float(pmax)))
        self.source_pub.publish(self._source_pose(now, source_x, source_y))
        self.grid_pub.publish(self._probability_grid(now, pmax))
        self.marker_pub.publish(self._markers(now, pmax, source_x, source_y))

    def _entropy(self) -> float:
        entropy = 0.0
        for probability in self.probabilities:
            if probability > 0.0:
                entropy -= probability * math.log(probability)
        return entropy

    def _source_estimate(self) -> Tuple[float, float, float]:
        max_index = max(range(self.cell_count), key=lambda index: self.probabilities[index])
        source_x, source_y = self._cell_center(max_index)
        return self.probabilities[max_index], source_x, source_y

    def _source_pose(self, stamp, source_x: float, source_y: float) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = source_x
        msg.pose.position.y = source_y
        msg.pose.orientation.w = 1.0
        return msg

    def _probability_grid(self, stamp, pmax: float) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.origin.position.x = self.map_min_x
        msg.info.origin.position.y = self.map_min_y
        msg.info.origin.orientation.w = 1.0

        scale = max(pmax, 1e-12)
        msg.data = [
            max(0, min(100, int(round(100.0 * probability / scale))))
            for probability in self.probabilities
        ]
        return msg

    def _cell_center(self, index: int) -> Tuple[float, float]:
        row = index // self.width
        col = index % self.width
        x = self.map_min_x + (col + 0.5) * self.resolution
        y = self.map_min_y + (row + 0.5) * self.resolution
        return x, y

    def _init_csv(self):
        directory = os.path.dirname(self.csv_output_file)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.csv_output_file, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "stamp_sec",
                "update_count",
                "detection",
                "robot_x",
                "robot_y",
                "wind_x",
                "wind_y",
                "entropy",
                "pmax",
                "source_estimate_x",
                "source_estimate_y",
            ])

    def _record_metrics(self, detected: bool):
        pmax, source_x, source_y = self._source_estimate()
        entropy = self._entropy()
        now = self.get_clock().now().nanoseconds * 1e-9
        robot_x, robot_y = self.robot_xy
        wind_x, wind_y = self.wind_xy
        with open(self.csv_output_file, "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                f"{now:.9f}",
                self.update_count,
                "hit" if detected else "void",
                f"{robot_x:.4f}",
                f"{robot_y:.4f}",
                f"{wind_x:.4f}",
                f"{wind_y:.4f}",
                f"{entropy:.6f}",
                f"{pmax:.9f}",
                f"{source_x:.4f}",
                f"{source_y:.4f}",
            ])

    def _markers(self, stamp, pmax: float, source_x: float, source_y: float) -> MarkerArray:
        markers = MarkerArray()
        markers.markers.append(self._source_estimate_marker(stamp, source_x, source_y))
        markers.markers.append(self._estimate_text_marker(stamp, pmax, source_x, source_y))
        return markers

    def _source_estimate_marker(self, stamp, source_x: float, source_y: float) -> Marker:
        marker = self._base_marker(stamp, 0, Marker.SPHERE)
        marker.pose.position.x = source_x
        marker.pose.position.y = source_y
        marker.pose.position.z = 0.45
        marker.scale.x = 0.30
        marker.scale.y = 0.30
        marker.scale.z = 0.30
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.25, a=0.95)
        return marker

    def _estimate_text_marker(self, stamp, pmax: float, source_x: float, source_y: float) -> Marker:
        marker = self._base_marker(stamp, 1, Marker.TEXT_VIEW_FACING)
        marker.pose.position.x = source_x
        marker.pose.position.y = source_y
        marker.pose.position.z = 0.85
        marker.scale.z = 0.28
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.25, a=0.95)
        marker.text = f"Pmax={pmax:.3f}"
        return marker

    def _base_marker(self, stamp, marker_id: int, marker_type: int) -> Marker:
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.frame_id
        marker.ns = "probability_map"
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker


def main(args=None):
    rclpy.init(args=args)
    node = ProbabilityMap()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
