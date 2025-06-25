#!/usr/bin/env python3

import os

import numpy as np
import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from normal_lane_following.msg import MultiMaskGroups
from std_msgs.msg import Bool, Float64, Int32MultiArray, String


class StopAtParkingLot(DTROS):
    def __init__(self, node_name):
        super(StopAtParkingLot, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle_name = os.environ.get("VEHICLE_NAME", "")
        self._node_active = False
        self.bridge = CvBridge()

        self.sub_control_mode = rospy.Subscriber(f"/{self._vehicle_name}/current_mode", Int32MultiArray, self.cbControlMode, queue_size=1)

        self.sub_masks = rospy.Subscriber(f"/{self._vehicle_name}/detect/masks", MultiMaskGroups, self.mask_callback, queue_size=1)

        self.pub_halt = rospy.Publisher(f"/{self._vehicle_name}/stopped_at_parkinglot", Bool, queue_size=1)
        self.pub_cmd_vel = rospy.Publisher(f"/{self._vehicle_name}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)

        rospy.loginfo("[DottedMaskChecker] Node gestartet")

    def cbControlMode(self, msg: Int32MultiArray):
        if msg.data[9] == 1:
            self._node_active = True
            rospy.loginfo(f"{self._vehicle_name}: Parking check active")
        else:
            self._node_active = False

    def mask_callback(self, msg: MultiMaskGroups):
        if not self._node_active:
            return

        # Konvertiere dotted-Masken
        dotted_masks = [self.bridge.imgmsg_to_cv2(m, desired_encoding="mono8") for m in msg.dotted]

        # Prüfe ob keine dotted_mask erkannt wurde
        dotted_missing = len(dotted_masks) == 0

        if dotted_missing == True:
            self.send_stop_cmd()

        # Publiziere Halt-Signal
        self.pub_halt.publish(Bool(dotted_missing))

        rospy.loginfo(f"[DottedMaskChecker] Dotted missing: {dotted_missing}")

    def send_stop_cmd(self):
        twist = Twist2DStamped()
        twist.v = 0.0
        twist.omega = 0.0
        self.pub_cmd_vel.publish(twist)

        self.pub_halt.publish(Bool(True))


if __name__ == "__main__":
    node = StopAtParkingLot(node_name="stop_if_no_dotted_mask_node")
    rospy.spin()
