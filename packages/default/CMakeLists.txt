cmake_minimum_required(VERSION 3.0.2)
project(default)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  sensor_msgs
  message_generation
)

## Message files
add_message_files(
  FILES
  BoundingBox.msg
  BoundingBoxArray.msg
)

## Generate added messages
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

## Declare catkin package
catkin_package(
  CATKIN_DEPENDS roscpp std_msgs geometry_msgs sensor_msgs message_runtime
)

## Include directories
include_directories(
  ${catkin_INCLUDE_DIRS}
)
