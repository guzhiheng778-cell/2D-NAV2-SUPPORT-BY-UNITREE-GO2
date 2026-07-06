from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_points_file = PathJoinSubstitution(
        [FindPackageShare("go2_config"), "config/autonomy", "patrol_points.yaml"]
    )
    default_params_file = PathJoinSubstitution(
        [FindPackageShare("go2_config"), "config/autonomy", "source_seek_greedy.yaml"]
    )
    default_v_info_arbiter_params_file = PathJoinSubstitution(
        [FindPackageShare("go2_config"), "config/autonomy", "v_info_arbiter.yaml"]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "points_file",
            default_value=default_points_file,
            description="YAML file containing patrol manager parameters and points",
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
        DeclareLaunchArgument(
            "stop_on_source_reached",
            default_value="true",
            description="Stop patrol permanently after source_seek reports source_reached",
        ),
        DeclareLaunchArgument(
            "optimize_resume_after_source_seek",
            default_value="true",
            description="Skip ahead to a nearer upcoming patrol point after source seeking",
        ),
        DeclareLaunchArgument(
            "resume_skip_lookahead",
            default_value="2",
            description="Number of upcoming patrol points to compare after source seeking",
        ),
        DeclareLaunchArgument(
            "resume_skip_distance_margin",
            default_value="0.25",
            description="Required distance improvement before skipping the current patrol point",
        ),
        DeclareLaunchArgument(
            "resume_skip_arrival_tolerance",
            default_value="0.45",
            description="Skip to an upcoming point immediately if already this close",
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params_file,
            description="YAML file containing greedy source seeking parameters",
        ),
        DeclareLaunchArgument(
            "v_info_arbiter_params_file",
            default_value=default_v_info_arbiter_params_file,
            description="YAML file containing v_info arbiter parameters",
        ),
        DeclareLaunchArgument(
            "enable_v_info_arbiter",
            default_value="true",
            description="Enable temporary /v_info takeover state machine",
        ),
        DeclareLaunchArgument(
            "enable_patrol_manager",
            default_value="true",
            description="Enable patrol waypoint manager",
        ),
        DeclareLaunchArgument(
            "enable_greedy_nav2_seek",
            default_value="false",
            description="Enable older 5.1 Nav2 source_estimate seeking node",
        ),
        Node(
            package="go2_config",
            executable="patrol_manager_node.py",
            name="patrol_manager",
            output="screen",
            condition=IfCondition(LaunchConfiguration("enable_patrol_manager")),#读取launch中的该参数，如果为true则运行该节点
            parameters=[
                {"points_file": LaunchConfiguration("points_file")},
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
                {"goal_settle_time_sec": LaunchConfiguration("goal_settle_time_sec")},
                {"goal_verify_tolerance": LaunchConfiguration("goal_verify_tolerance")},
                {"goal_verify_yaw_tolerance": LaunchConfiguration("goal_verify_yaw_tolerance")},
                {"max_goal_retries": LaunchConfiguration("max_goal_retries")},
                {"stop_on_source_reached": LaunchConfiguration("stop_on_source_reached")},
                {
                    "optimize_resume_after_source_seek": LaunchConfiguration(
                        "optimize_resume_after_source_seek"
                    )
                },
                {"resume_skip_lookahead": LaunchConfiguration("resume_skip_lookahead")},
                {
                    "resume_skip_distance_margin": LaunchConfiguration(
                        "resume_skip_distance_margin"
                    )
                },
                {
                    "resume_skip_arrival_tolerance": LaunchConfiguration(
                        "resume_skip_arrival_tolerance"
                    )
                },
            ],
        ),
        Node(
            package="go2_config",
            executable="source_seek_greedy_node.py",
            name="source_seek_greedy",
            output="screen",
            condition=IfCondition(LaunchConfiguration("enable_greedy_nav2_seek")),
            parameters=[
                LaunchConfiguration("params_file"),
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
            ],
        ),
        Node(
            package="go2_config",
            executable="v_info_arbiter_node.py",
            name="v_info_arbiter",
            output="screen",
            condition=IfCondition(LaunchConfiguration("enable_v_info_arbiter")),
            parameters=[
                LaunchConfiguration("v_info_arbiter_params_file"),
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
            ],
        ),
    ])
