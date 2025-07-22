#!/usr/bin/env python3

import os

import rospy
import numpy as np
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from normal_lane_following.msg import MultiMaskGroups
from std_msgs.msg import Bool, Float64, Int32MultiArray, String
from duckietown_msgs.msg import Twist2DStamped




class LeaveLane:
    def __init__(self):
        rospy.init_node("dotted_mask_checker")

        self._vehicle_name = os.environ.get("VEHICLE_NAME", "")
        self._node_active = False        
        self.bridge = CvBridge()

        self.sub_control_mode = rospy.Subscriber(f"/{self._vehicle_name}/current_mode", Int32MultiArray, self.cbControlMode, queue_size=1)

        self.laneleft = rospy.Publisher(f"/{self._vehicle_name}/Change/LaneLeft", Bool, queue_size=1 )

        rospy.loginfo("[DottedMaskChecker] Node gestartet")


    def cbControlMode(self, msg: Int32MultiArray):
        if msg.data[8] == 1:
            self._node_active = True
            rospy.loginfo(f"{self._vehicle_name}: Parking check active")
        else:
            self._node_active = False


    def leavingLane(self):
            # hard coded transition to opposite lane following
            v = 0.35
            omega = 5  # rad/s nach links (negativ)
            duration = 1  # math.pi / (2 * abs(omega))  # Zeit für 90° Drehung
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
    
