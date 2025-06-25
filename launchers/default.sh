#!/bin/bash

source /environment.sh

cd /code/catkin_ws
catkin build
source devel/setup.bash


cd /code/catkin_ws
catkin clean default
catkin build default
source devel/setup.bash

# initialize launch file
dt-launchfile-init

# Check CUDA availability
echo "Checking CUDA availability..."
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Device Count:', torch.cuda.device_count()); print('CUDA Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A');" || echo "Error checking CUDA"


# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------


# NOTE: Use the variable DT_REPO_PATH to know the absolute path to your code
# NOTE: Use `dt-exec COMMAND` to run the main process (blocking process)

# launching app
# dt-exec python3 -m "my_package.my_script"


rosrun default LaneSegmentation.py &
# rosrun default ObjectDetection.py &
rosrun normal_lane_following NormalLaneFollowing.py &
rosrun normal_lane_following OppositeLaneFollowing.py &
rosrun normal_lane_following DuckieCenterCheck.py &
rosrun normal_lane_following DuckieCheckRight.py &
rosrun intersection_handling IntersectionHandling.py &
rosrun intersection_handling IntersectionDetection.py &
rosrun intersection_handling StopAtIntersection.py &
rosrun parking ParkStopGo.py &
rosrun parking SearchForParkingLot.py &
rosrun parking StopAtParkingLot.py &
# rosrun duckie_control challenge_test_publisher.py &
rosrun duckie_control PIDControlLane.py &
rosrun duckie_control SwitchControlNode.py &







#dt-exec python3 "${DT_REPO_PATH}/packages/followlane/src/camera_reader_node.py" &
#dt-exec python3 "${DT_REPO_PATH}/packages/followlane/src/control_lane_node.py" &
#dt-exec python3 "${DT_REPO_PATH}/packages/followlane/src/self_detect_lane_node.py" 
#dt-exec python3 "${DT_REPO_PATH}/packages/followlane/src/camera_reader_node.py" 
#dt-exec python3 "${DT_REPO_PATH}/packages/followlane/src/yolo_lane_detection.py"



# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join
