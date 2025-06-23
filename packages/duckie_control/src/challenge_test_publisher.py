#!/usr/bin/env python3

import os
import sys
from enum import Enum

import rospy
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import Int32MultiArray, MultiArrayDimension


class ControlMode(Enum):
    NormalLaneFollowing = [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0]
    AvoidDuckies = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    StoppingAtIntersection = [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]
    IntersectionHandling = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    SearchForParkingLot = [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
    StopAtParkingLot = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
    ParkingManager = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]


class ChallengeTestPublisher(DTROS):  # FIX: Add inheritance from DTROS
    def __init__(self, node_name):
        super(ChallengeTestPublisher, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle_name = os.environ["VEHICLE_NAME"]

        # Create a publisher for the Int32MultiArray message type
        self.pub = rospy.Publisher(f"/{self._vehicle_name}/current_mode", Int32MultiArray, queue_size=1, latch=True)

        # Default challenge mode
        self.current_mode = ControlMode.IntersectionHandling

        # Publishing rate (Hz)
        self.rate = rospy.Rate(10)

    def publish_challenge_mode(self):
        msg = Int32MultiArray()
        msg.data = self.current_mode.value
        self.pub.publish(msg)
        rospy.loginfo(f"Publishing challenge mode: {self.current_mode} - {msg.data}")

    def start_publishing(self):
        while not rospy.is_shutdown():
            self.publish_challenge_mode()
            self.rate.sleep()


if __name__ == "__main__":
    node = ChallengeTestPublisher(node_name="challenge_test_publisher")
    node.start_publishing()  # FIX: Added parentheses to call the method
