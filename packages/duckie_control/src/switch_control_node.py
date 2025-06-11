#!/usr/bin/env python3

import os
from enum import Enum

import rospy
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import Bool, Float64, Int32, String


class ControlType(Enum):
    WAIT = "wait"
    LANE = "lane"
    INTERSECTION = "intersection"
    DUCKIE = "duckie"
    DUCKIEBOT = "duckiebot"
    PARKINGSPOT = "parking_spot"


class SwitchControlNode(DTROS):
    def __init__(self, node_name):
        super(SwitchControlNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle_name = os.environ["VEHICLE_NAME"]

        # state initialization
        self._control_mode = ControlType.Lane
        self._previous_control_mode = None

        # Subscribers
        self.sub_lane = rospy.Subscriber(f"/{self._vehicle_name}/detect/lane", Float64, self.cbLaneDetected, queue_size=1)
        self.sub_red_stop = rospy.Subscriber(f"/{self._vehicle_name}/detect/red_stop", Bool, self.cbRedStopDetected, queue_size=1)
        self.sub_duckie = rospy.Subscriber(f"/{self._vehicle_name}/detect/duckie", Float64, self.cbDuckieDetected, queue_size=1)
        self.sub_duckiebot = rospy.Subscriber(f"/{self._vehicle_name}/detect/duckiebot", Float64, self.cbDuckiebotDetected, queue_size=1)
        self.sub_parking_spot = rospy.Subscriber(f"/{self._vehicle_name}/detect/parking_spot", Float64, self.cbParkingSpotDetected, queue_size=1)

        self.sub_intersection_done = rospy.Subscriber(f"/{self._vehicle_name}/intersection/done", Bool, self.cbIntersectionDone, queue_size=1)

        # Publishers
        self.pub_control = rospy.Publisher(f"/{self._vehicle_name}/switch/control", Int32, queue_size=1)
        self.pub_control_mode = rospy.Publisher(f"/{self._vehicle_name}/control_mode", String, queue_size=1)

        rospy.loginfo(f"{self._vehicle_name}: SwitchControlNode initialized with control mode: {self._control_mode}")

    def cbLaneDetected(self, msg):
        if self._control_mode == ControlType.Lane:
            print("Test")

    def cbRedStopDetected(self, msg):
        if msg.data and self._control_mode == ControlType.LANE:
            self.SwitchControlMode(ControlType.INTERSECTION)
            rospy.loginfo("Red stop detected, switching to intersection control")

    def cbIntersectionDone(self, msg):
        if msg.data and self._control_mode == ControlType.INTERSECTION:
            self.SwitchControlMode(ControlType.LANE)
            rospy.loginfo("Intersection action completed, switching back to lane control")

    def cbDuckieDetected(self, msg):
        # if msg.data and self._control_mode == ControlType.INTERSECTION:
        #     self.SwitchControlMode(ControlType.LANE)
        #     rospy.loginfo("Intersection action completed, switching back to lane control")
        print("Write your Code here like the above")

    def cbDuckiebotDetected(self, msg):
        # if msg.data and self._control_mode == ControlType.INTERSECTION:
        #     self.SwitchControlMode(ControlType.LANE)
        #     rospy.loginfo("Intersection action completed, switching back to lane control")
        print("Write your Code here like the above")

    def cbParkingSpotDetected(self, msg):
        # if msg.data and self._control_mode == ControlType.INTERSECTION:
        #     self.SwitchControlMode(ControlType.LANE)
        #     rospy.loginfo("Intersection action completed, switching back to lane control")
        print("Write your Code here like the above")

    def SwitchControlMode(self, control_mode):
        if control_mode in ControlType:
            self._previous_control_mode = self._control_mode
            self._control_mode = control_mode
            rospy.loginfo(f"Control mode switched to: {self._control_mode}")
            self.publish_control_mode()
        else:
            rospy.logwarn(f"Invalid control mode: {control_mode}")

    def run(self):
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            # publish the current control mode
            msg_control = Int32()
            msg_control.data = self._control_mode.value
            self.pub_control.publish(msg_control)

            rate.sleep()


if __name__ == "__main__":
    node = SwitchControlNode(node_name="switch_control_node")
    node.run()
    rospy.spin()
