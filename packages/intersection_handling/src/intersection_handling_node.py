#!/usr/bin/env python3

import os
import random
import time
from enum import Enum

import rospy
from duckie_control.src.switch_control_node import ControlType
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from std_msgs.msg import Float64, Int32


class IntersectionState(Enum):
    OFF = "off"
    STOPPING = "stopping"
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

        # Publishers
        self.pub_cmd_vel = rospy.Publisher(f"/{self._vehicle_name}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)
        self.pub_intersection_done = rospy.Publisher(f"/{self._vehicle_name}/intersection/done", Bool, queue_size=1)

        rospy.loginfo(f"{self._vehicle_name}: IntersectionHandlingNode initialized with state: {self._state}")

    def chooseIntersectionDirection(self):
        directions = [IntersectionDirection.LEFT, IntersectionDirection.STRAIGHT, IntersectionDirection.RIGHT]
        self._intersection_direction = random.choice(directions)

        action_msg = String()
        directions_msg = self._intersection_direction.value
        self.pub_intersection_direction.publish(directions_msg)
        rospy.loginfo(f"Chosen intersection direction: {self._intersection_direction}")
