#!/usr/bin/env python3

import os
import random
import time
from enum import Enum

import rospy
from duckie_control.src.switch_control_node import ControlType
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from normal_lane_following.msg import MultiMaskGroups
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Float64, Int32, String


class IntersectionState(Enum):
    STOPPING = "stopped"
    CHOOSING_ACTION = "choosing_action"
    EXECUTING_ACTION = "executing_action"
    COMPLETED = "completed"


class IntersectionDirection(Enum):
    LEFT = "left"
    STRAIGHT = "straight"
    RIGHT = "right"


class IntersectionHandlingNode(DTROS):
    def __init__(self, node_name):
        super(IntersectionHandlingNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle_name = os.environ["VEHICLE_NAME"]

        # State initialization
        self._active = False
        self._state = IntersectionState.OFF
        self._intersection_direction = None
        self._action_start_time = None

        # Subscribers
        self.sub_control_mode = rospy.Subscriber(f"/{self._vehicle_name}/switch/control", Int32, self.cbControlMode, queue_size=1)
        self.sub_lane = rospy.Subscriber(f"/{self._vehicle_name}/detect/lane", Float64, self.cbLaneDetected, queue_size=1)
        self.sub_masks = rospy.Subscriber(f"/{self._vehicle_name}/detect/masks", String, self.cbMasksDetected, queue_size=1)

        # Publishers
        self.pub_cmd_vel = rospy.Publisher(f"/{self._vehicle_name}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)
        self.pub_intersection_done = rospy.Publisher(f"/{self._vehicle_name}/intersection/done", Bool, queue_size=1)

        rospy.loginfo(f"{self._vehicle_name}: IntersectionHandlingNode initialized with state: {self._state}")

    def cbmasks(self, msg: MultiMaskGroups):
        # Wandelt sensor_msgs/Image[] in OpenCV-Bilder um
        # self.white_masks = [self.bridge.imgmsg_to_cv2(m, desired_encoding="mono8") for m in msg.white]
        # self.yellow_masks = [self.bridge.imgmsg_to_cv2(m, desired_encoding="mono8") for m in msg.yellow]
        self.red_masks = [self.bridge.imgmsg_to_cv2(m, desired_encoding="mono8") for m in msg.red]
        # self.dotted_masks = [self.bridge.imgmsg_to_cv2(m, desired_encoding="mono8") for m in msg.dotted]

    def chooseIntersectionDirection(self):
        intersection_type = ["LeftStraightRight", "LeftStraight", "LeftRight", "StraightRight"]

        # TODO: Integrate logic to determine the intersection type
        if "number of red masks" == 4:
            intersection_type = "LeftStraightRight"
        elif "number of red masks" == 3:
            print(" ")

        directions = []
        if intersection_type == "LeftStraightRight":
            directions = [IntersectionDirection.LEFT, IntersectionDirection.STRAIGHT]
        elif intersection_type == "LeftStraight":
            directions = [IntersectionDirection.LEFT, IntersectionDirection.STRAIGHT]
        elif intersection_type == "LeftRight":
            directions = [IntersectionDirection.LEFT, IntersectionDirection.RIGHT]
        elif intersection_type == "StraightRight":
            directions = [IntersectionDirection.STRAIGHT, IntersectionDirection.RIGHT]

        if not directions == []:
            self._intersection_direction = random.choice(directions)
            rospy.loginfo(f"Chosen intersection direction: {self._intersection_direction}")
            self.turn()
        else:
            rospy.logwarn("No valid intersection direction found, defaulting to STRAIGHT")
            self._intersection_direction = IntersectionDirection.STRAIGHT

    def turn(self):
        match self._intersection_direction:
            case IntersectionDirection.LEFT:
                rospy.loginfo("Turning left")
                # TODO: Go straight for a bit, then turn left
            case IntersectionDirection.STRAIGHT:
                rospy.loginfo("Going straight")
                # TODO: Go straight for a defined distance
            case IntersectionDirection.RIGHT:
                rospy.loginfo("Turning right")
                # TODO: Go straight for a bit, then turn right


if __name__ == "__main__":
    # create the node
    node = IntersectionHandlingNode(node_name="intersection_handling_node")
    # node.run()
    # keep the process from terminating
    rospy.spin()
