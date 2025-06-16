#!/usr/bin/env python3

import os

import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from std_msgs.msg import Float64, Int32
from switch_control_node import ControlType


class IntersectionHandlingNode(DTROS):
    def __init__(self, node_name):
        super(IntersectionHandlingNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

    def test(self):
        print("test")


if __name__ == "__main__":
    # create the node
    node = IntersectionHandlingNode(node_name="intersection_handling_node")
    node.run()
    # keep the process from terminating
    rospy.spin()
