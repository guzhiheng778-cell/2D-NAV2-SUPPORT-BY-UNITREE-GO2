from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    go2_config_share = FindPackageShare("go2_config")

    gazebo_launch = PathJoinSubstitution(
        [go2_config_share, "launch", "gazebo_velodyne.launch.py"]
    )
    navigate_launch = PathJoinSubstitution(
        [go2_config_share, "launch", "navigate.launch.py"]
    )
    odor_probability_launch = PathJoinSubstitution(
        [go2_config_share, "launch", "odor_probability.launch.py"]
    )
    local_info_gain_launch = PathJoinSubstitution(
        [go2_config_share, "launch", "local_info_gain.launch.py"]
    )

    default_world = PathJoinSubstitution(
        [go2_config_share, "worlds", "playground.world"]
    )
    default_map = PathJoinSubstitution(
        [go2_config_share, "maps", "playground.yaml"]
    )
    default_nav_params = PathJoinSubstitution(
        [go2_config_share, "config/autonomy", "navigation.yaml"]
    )
    default_odor_params = PathJoinSubstitution(
        [go2_config_share, "config/autonomy", "odor_simulation.yaml"]
    )
    default_probability_params = PathJoinSubstitution(
        [go2_config_share, "config/autonomy", "probability_map.yaml"]
    )
    default_local_info_params = PathJoinSubstitution(
        [go2_config_share, "config/autonomy", "local_info_gain.yaml"]
    )
    default_rviz_config = PathJoinSubstitution(
        [FindPackageShare("champ_navigation"), "rviz", "navigation_optimized.rviz"]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation clock for all nodes",
        ),
        DeclareLaunchArgument(
            "world",
            default_value=default_world,
            description="Gazebo world file",
        ),
        DeclareLaunchArgument(
            "map",
            default_value=default_map,
            description="Nav2 map YAML file",
        ),
        DeclareLaunchArgument(
            "nav_params_file",
            default_value=default_nav_params,
            description="Nav2 parameter YAML file",
        ),
        DeclareLaunchArgument(
            "odor_params_file",
            default_value=default_odor_params,
            description="Wind and odor simulation parameter YAML file",
        ),
        DeclareLaunchArgument(
            "probability_params_file",
            default_value=default_probability_params,
            description="Probability map parameter YAML file",
        ),
        DeclareLaunchArgument(
            "local_info_params_file",
            default_value=default_local_info_params,
            description="Local information gain parameter YAML file",
        ),
        DeclareLaunchArgument(
            "nav_rviz",
            default_value="true",
            description="Start RViz with the optimized navigation view",
        ),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=default_rviz_config,
            description="RViz config file",
        ),
        DeclareLaunchArgument(
            "gazebo_gui",
            default_value="true",
            description="Start Gazebo GUI",
        ),
        DeclareLaunchArgument(
            "local_info_gain",
            default_value="true",
            description="Start local information gain node publishing /v_info",
        ),
        DeclareLaunchArgument("world_init_x", default_value="0.0"),
        DeclareLaunchArgument("world_init_y", default_value="0.0"),
        DeclareLaunchArgument("world_init_z", default_value="0.6"),
        DeclareLaunchArgument("world_init_heading", default_value="0.0"),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gazebo_launch),
            launch_arguments={
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "world": LaunchConfiguration("world"),
                "gui": LaunchConfiguration("gazebo_gui"),
                "rviz": "false",
                "world_init_x": LaunchConfiguration("world_init_x"),
                "world_init_y": LaunchConfiguration("world_init_y"),
                "world_init_z": LaunchConfiguration("world_init_z"),
                "world_init_heading": LaunchConfiguration("world_init_heading"),
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(navigate_launch),
            launch_arguments={
                "map": LaunchConfiguration("map"),
                "params_file": LaunchConfiguration("nav_params_file"),
                "sim": LaunchConfiguration("use_sim_time"),
                "rviz": "false",
            }.items(),
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", LaunchConfiguration("rviz_config")],
            condition=IfCondition(LaunchConfiguration("nav_rviz")),
            parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(odor_probability_launch),
            launch_arguments={
                "odor_params_file": LaunchConfiguration("odor_params_file"),
                "probability_params_file": LaunchConfiguration("probability_params_file"),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(local_info_gain_launch),
            condition=IfCondition(LaunchConfiguration("local_info_gain")),
            launch_arguments={
                "params_file": LaunchConfiguration("local_info_params_file"),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
            }.items(),
        ),
    ])
