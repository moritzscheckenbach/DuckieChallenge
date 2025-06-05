#!/usr/bin/env python3

import os

import rospy
from duckie_control.src.switch_control_node import ControlType
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from std_msgs.msg import Float64, Int32


class IntersectionHandlingNode(DTROS):
    def __init__(self, node_name):
        super(IntersectionHandlingNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
