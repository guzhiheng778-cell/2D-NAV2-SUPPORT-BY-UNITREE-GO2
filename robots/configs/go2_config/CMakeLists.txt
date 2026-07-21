cmake_minimum_required(VERSION 3.5)
project(go2_config)

find_package(ament_cmake REQUIRED)

install(
  PROGRAMS
    scripts/odor_simulator_node.py
    scripts/infotaxis_node.py
    scripts/local_info_gain_node.py
    scripts/patrol_manager_node.py
    scripts/probability_map_node.py
    scripts/source_seek_greedy_node.py
    scripts/v_info_arbiter_node.py
    scripts/wind_simulator_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config maps worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
