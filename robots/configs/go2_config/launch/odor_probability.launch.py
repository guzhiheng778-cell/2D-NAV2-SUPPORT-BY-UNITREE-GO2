from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_odor_params_file = PathJoinSubstitution(
        [FindPackageShare("go2_config"), "config/autonomy", "odor_simulation.yaml"]
    )
    default_probability_params_file = PathJoinSubstitution(
        [FindPackageShare("go2_config"), "config/autonomy", "probability_map.yaml"]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "odor_params_file",
            default_value=default_odor_params_file,
            description="YAML file containing wind and odor simulation parameters",
        ),
        DeclareLaunchArgument(
            "probability_params_file",
            default_value=default_probability_params_file,
            description="YAML file containing probability map parameters",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation clock",
        ),
        Node(
            package="go2_config",
            executable="wind_simulator_node.py",
            name="wind_simulator",
            output="screen",
            parameters=[
                LaunchConfiguration("odor_params_file"),
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
            ],
        ),
        Node(
            package="go2_config",
            executable="odor_simulator_node.py",
            name="odor_simulator",
            output="screen",
            parameters=[
                LaunchConfiguration("odor_params_file"),
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
            ],
        ),
        Node(
            package="go2_config",
            executable="probability_map_node.py",
            name="probability_map",
            output="screen",
            parameters=[
                LaunchConfiguration("probability_params_file"),
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
            ],
        ),
    ])
