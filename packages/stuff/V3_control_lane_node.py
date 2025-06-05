#!/usr/bin/env python3

import os

import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from std_msgs.msg import Float64, Int32
from switch_control_node import ControlType


class ControlLaneNode(DTROS):
    def __init__(self, node_name):
        super(ControlLaneNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self.enable = False
        self._vehicle_name = os.environ["VEHICLE_NAME"]
        twist_topic = f"/{self._vehicle_name}/car_cmd_switch_node/cmd"
        self.pub_cmd_vel = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)

        self.sub_lane = rospy.Subscriber(f"/{self._vehicle_name}/detect/lane", Float64, self.cbFollowLane, queue_size=1)
        self.sub_control = rospy.Subscriber(f"/{self._vehicle_name}/switch/control", Int32, self.cbControl, queue_size=1)

        # PID Parameter
        self.k_p = rospy.get_param("~k_p", 1.0)
        self.k_i = rospy.get_param("~k_i", 0.0)
        self.k_d = rospy.get_param("~k_d", 0.0)
        self.base_speed = rospy.get_param("~base_speed", 0.2)
        self.image_center = 500  # pixel (wird erwartet vom Lane-Node)

        self.integral = 0.0
        self.last_error = 0.0

        rospy.on_shutdown(self.fnShutDown)

    def cbControl(self, msg):
        self.enable = msg.data == ControlType.Lane.value

    def cbFollowLane(self, desired_center):
        if not self.enable:
            return

        center = desired_center.data
        self.followLane(center)

    def followLane(self, center):
        # Fehlerberechnung
        error = (center - self.image_center) / 100.0  # normalisiert (optional)

        # PID
        self.integral += error
        derivative = error - self.last_error
        self.last_error = error

        omega = self.k_p * error + self.k_i * self.integral + self.k_d * derivative

        twist = Twist2DStamped(v=self.base_speed, omega=omega)
        rospy.loginfo(f"[PID] error={error:.3f} | v={self.base_speed:.2f}, omega={omega:.3f}")
        self.pub_cmd_vel.publish(twist)

    def fnShutDown(self):
        rospy.loginfo("Shutting down. cmd_vel will be 0")
        twist = Twist2DStamped(v=0.0, omega=0.0)
        self.pub_cmd_vel.publish(twist)


if __name__ == "__main__":
    node = ControlLaneNode(node_name="control_lane_node")
    rospy.spin()
