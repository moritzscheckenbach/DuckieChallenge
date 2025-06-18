#!/bin/bash

source /environment.sh

cd /code/catkin_ws
catkin build
source devel/setup.bash

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------


# NOTE: Use the variable DT_REPO_PATH to know the absolute path to your code
# NOTE: Use `dt-exec COMMAND` to run the main process (blocking process)

# launching app
# dt-exec python3 -m "my_package.my_script"


rosrun default LaneSegmentation.py &
rosrun default ObjectDetection.py &
rosrun normal_lane_following NormalLaneFollowing.py &
rosrun normal_lane_following OppositeLaneFollowing.py &
rosrun normal_lane_following DuckieCenterCheck.py &
rosrun duckie_control PID_control_lane_node.py &
rosrun intersection_handling intersection_handling_node.py &
rosrun duckie_control switch_control_node.py &






#dt-exec python3 "${DT_REPO_PATH}/packages/followlane/src/camera_reader_node.py" &
#dt-exec python3 "${DT_REPO_PATH}/packages/followlane/src/control_lane_node.py" &
#dt-exec python3 "${DT_REPO_PATH}/packages/followlane/src/self_detect_lane_node.py" 
#dt-exec python3 "${DT_REPO_PATH}/packages/followlane/src/camera_reader_node.py" 
#dt-exec python3 "${DT_REPO_PATH}/packages/followlane/src/yolo_lane_detection.py"




# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join
