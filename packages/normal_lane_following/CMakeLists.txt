cmake_minimum_required(VERSION 2.8.3)
project(normal_lane_following)

find_package(catkin REQUIRED COMPONENTS
  rospy
  message_generation
  sensor_msgs
)

add_message_files(
  FILES
  MultiMaskGroups.msg
)

generate_messages(
  DEPENDENCIES
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime rospy sensor_msgs
)