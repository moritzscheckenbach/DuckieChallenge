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

        rospy.on_shutdown(self.fnShutDown)

    def cbControl(self, msg):
        if msg.data == ControlType.Lane.value:
            self.enable = True

        else:
            self.enable = False

    def cbFollowLane(self, desired_center):

        print(f"received message. enabled : {self.enable}")

        if not self.enable:
            return

        center = desired_center.data
        self.followLane(center)

    def followLane(self, center):
        """
        PID controller for lane following

        Args:
            center: Position of lane center in pixels
        """

        # PID Parameters
        Kp = 0.5  # Proportional gain
        Ki = 0.0001  # Integral gain
        Kd = 0.1  # Derivative gain

        # Initialize PID variables if not already set
        if not hasattr(self, "prev_error"):
            self.prev_error = 0
            self.integral = 0

        # Calculate current error
        current_error = (center - 320) / 100

        # Calculate integral term with anti-windup
        self.integral += current_error
        if self.integral > 100:  # Limit integral windup
            self.integral = 100
        elif self.integral < -100:
            self.integral = -100

        # Calculate derivative term
        derivative = current_error - self.prev_error

        # Save current error for next iteration
        self.prev_error = current_error

        # Calculate PID output
        pid_output = Kp * current_error + Ki * self.integral + Kd * derivative

        # Limit the steering angle
        if pid_output > 8.0:
            pid_output = 8.0
        elif pid_output < -8.0:
            pid_output = -8.0

        # Adjust velocity based on curve sharpness (slow down in curves)
        base_speed = 0.3
        curve_factor = abs(pid_output) / 8.0  # Normalized curve sharpness
        v = base_speed * (1.0 - 0.5 * curve_factor)  # Reduce speed in curves

        twist = Twist2DStamped(v=v, omega=pid_output)
        print(f"moving {v} with omega {pid_output} at error {current_error}")
        self.pub_cmd_vel.publish(twist)

    def fnShutDown(self):
        rospy.loginfo("Shutting down. cmd_vel will be 0")

        twist = Twist2DStamped(v=0.0, omega=0.0)
        self.pub_cmd_vel.publish(twist)


if __name__ == "__main__":
    # create the node
    node = ControlLaneNode(node_name="control_lane_node")
    # keep the process from terminating
    rospy.spin()
