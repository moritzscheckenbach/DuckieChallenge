#!/usr/bin/env python3

import os
import time

import numpy as np
import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from normal_lane_following.msg import MultiMaskGroups
from std_msgs.msg import Bool, Float64, Int32MultiArray, String


class LeaveLane(DTROS):
    def __init__(self, node_name):
        super(LeaveLane, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle_name = os.environ.get("VEHICLE_NAME", "")
        self._node_active = False
        self.bridge = CvBridge()

        self.sub_control_mode = rospy.Subscriber(f"/{self._vehicle_name}/current_mode", Int32MultiArray, self.cbControlMode, queue_size=1)
        self.pub_cmd_vel = rospy.Publisher(f"/{self._vehicle_name}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)

        self.laneleft = rospy.Publisher(f"/{self._vehicle_name}/Change/LaneLeft", Bool, queue_size=1)

        rospy.logwarn("[DottedMaskChecker] Node gestartet")

    def cbControlMode(self, msg: Int32MultiArray):
        if msg.data[14] == 1:
            self._node_active = True
            rospy.logwarn(f"{self._vehicle_name}: Leaving Lane")
            self.leavingLane()
        else:
            self._node_active = False

    def leavingLane(self):
        # hard coded transition to opposite lane following
        rospy.logwarn(f"{self._vehicle_name}: Leaving Lane - Going to opposite lane")
        v = 0.4
        omega = 35  # rad/s nach links (negativ)
        duration = 0.35  # math.pi / (2 * abs(omega))  # Zeit für 90° Drehung
        rate = rospy.Rate(10)
        start_time = time.time()

        cmd_msg = Twist2DStamped()
        cmd_msg.v = v
        cmd_msg.omega = omega

        while time.time() - start_time < duration:
            self.pub_cmd_vel.publish(cmd_msg)
            rate.sleep()

        self.laneleft.publish(Bool(True))


if __name__ == "__main__":
    node = LeaveLane(node_name="LeaveLane")
    rospy.spin()
