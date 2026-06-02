from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_points_file = PathJoinSubstitution(
        [FindPackageShare("go2_config"), "config/autonomy", "patrol_points.yaml"]
    )
    default_source_seek_params_file = PathJoinSubstitution(
        [FindPackageShare("go2_config"), "config/autonomy", "source_seek_greedy.yaml"]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "points_file",
            default_value=default_points_file,
            description="YAML file containing patrol manager parameters and points",
        ),
        DeclareLaunchArgument(
            "source_seek_params_file",
            default_value=default_source_seek_params_file,
            description="YAML file containing greedy source seeking parameters",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation clock",
        ),
        DeclareLaunchArgument(
            "goal_settle_time_sec",
            default_value="2.0",
            description="Extra time to let localization settle before verifying a reached goal",
        ),
        DeclareLaunchArgument(
            "goal_verify_tolerance",
            default_value="0.30",
            description="Maximum verified XY distance from the active patrol goal",
        ),
        DeclareLaunchArgument(
            "goal_verify_yaw_tolerance",
            default_value="0.60",
            description="Maximum verified yaw error from the active patrol goal",
        ),
        DeclareLaunchArgument(
            "max_goal_retries",
            default_value="2",
            description="Number of times to resend a goal after verification fails",
        ),
        Node(
            package="go2_config",
            executable="patrol_manager_node.py",
            name="patrol_manager",
            output="screen",
            parameters=[
                {"points_file": LaunchConfiguration("points_file")},
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
                {"goal_settle_time_sec": LaunchConfiguration("goal_settle_time_sec")},
                {"goal_verify_tolerance": LaunchConfiguration("goal_verify_tolerance")},
                {"goal_verify_yaw_tolerance": LaunchConfiguration("goal_verify_yaw_tolerance")},
                {"max_goal_retries": LaunchConfiguration("max_goal_retries")},
            ],
        ),
        Node(
            package="go2_config",
            executable="source_seek_greedy_node.py",
            name="source_seek_greedy",
            output="screen",
            parameters=[
                LaunchConfiguration("source_seek_params_file"),
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
            ],
        ),
    ])
