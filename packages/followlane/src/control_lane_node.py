#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64, Int32

from duckietown_msgs.msg import Twist2DStamped
import os
from duckietown.dtros import DTROS, NodeType
from switch_control_node import ControlType

class ControlLaneNode(DTROS):
    def __init__(self,node_name):
        super(ControlLaneNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        
        self.enable = False
        self._vehicle_name = os.environ['VEHICLE_NAME']
        twist_topic = f"/{self._vehicle_name}/car_cmd_switch_node/cmd"
        self.pub_cmd_vel = rospy.Publisher(twist_topic, Twist2DStamped, queue_size = 1)

        self.sub_lane = rospy.Subscriber(f'/{self._vehicle_name}/detect/lane', Float64, self.cbFollowLane, queue_size = 1)
        self.sub_control = rospy.Subscriber(f"/{self._vehicle_name}/switch/control", Int32, self.cbControl , queue_size = 1)
        
        
        rospy.on_shutdown(self.fnShutDown)

    def cbControl(self,msg):
        if msg.data == ControlType.Lane.value:
            self.enable = True
        
        else:
            self.enable = False

    def cbFollowLane(self, desired_center):

        print(f'received message. enabled : {self.enable}')

        if not self.enable:
            return        
        
        center = desired_center.data
        self.followLane(center)

    def followLane(self, center):
        # Write your code for a PID controller here
        error = (center - 500) / 100

        v = 0.2
        a = error
        
        twist = Twist2DStamped(v=v, omega=a)
        print(f'moving {v} {a} error {error}')
        self.pub_cmd_vel.publish(twist)

    def fnShutDown(self):
        rospy.loginfo("Shutting down. cmd_vel will be 0")

        twist = Twist2DStamped(v=0.0, omega=0.0)
        self.pub_cmd_vel.publish(twist) 

if __name__ == '__main__':
    # create the node
    node = ControlLaneNode(node_name='control_lane_node')
    # keep the process from terminating
    rospy.spin()