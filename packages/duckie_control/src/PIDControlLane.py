#!/usr/bin/env python3

import os

import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from std_msgs.msg import Float64, Int32, Int32MultiArray


class ControlLaneNode(DTROS):
    def __init__(self, node_name):
        super(ControlLaneNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle_name = os.environ["VEHICLE_NAME"]

        self._mode_topic = f"/{self._vehicle_name}/current_mode"
        self.sub_modus = rospy.Subscriber(self._mode_topic, Int32MultiArray, self.cbControlMode, queue_size=1)

        twist_topic = f"/{self._vehicle_name}/car_cmd_switch_node/cmd"
        self.pub_cmd_vel = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)

        self.sub_lane = rospy.Subscriber(f"/{self._vehicle_name}/detect/lane", Float64, self.cbFollowLane, queue_size=1)

        self._node_active = False

        rospy.on_shutdown(self.fnShutDown)

    def cbControlMode(self, msg: Int32MultiArray):
        if msg.data[4] == 1:
            self._node_active = True
            rospy.logwarn(f"{self._vehicle_name}: PID control is now active")
        else:
            self._node_active = False
            self.fnShutDown()

        if (msg.data[8] == 1) or (msg.data[2] == 1) or (msg.data[9] == 1):
            # PID Parameters
            # Kp = 4.00  # Proportional gain
            # Ki = 0.07  # Integral gain
            # Kd = 4.50  # Derivative gain

            # # Sehr Gute Werte für GUSTAV am 25.06.2025
            # Kp = 9.80  # Proportional gain
            # Ki = 0.04  # Integral gain
            # Kd = 0.90  # Derivative gain

            # # Sehr Gute Werte für Dorette - Auf ersten Videos zu sehen
            # Kp = 9.80  # Proportional gain
            # Ki = 0.04  # Integral gain
            # Kd = 0.85  # Derivative gain

            # Kp = 7.80  # Proportional gain
            # Ki = 0.075  # Integral gain
            # Kd = 1.25  # Derivative gain

            self.base_speed = 0.55
            self.Kp = 9.92  # Proportional gain
            self.Ki = 0.18  # Integral gain
            self.Kd = 0.65  # Derivative gain
        else:
            self.base_speed = 0.35
            self.Kp = 15.00  # Proportional gain
            self.Ki = 0.55  # Integral gain
            self.Kd = 0.60  # Derivative gain

    def cbFollowLane(self, desired_center):

        # rospy.logwarn(f"received message. enabled : {self._node_active}")

        if not self._node_active:
            return

        center = desired_center.data
        self.followLane(center)

    def followLane(self, center):
        """
        PID controller for lane following

        Args:
            center: Position of lane center in pixels
        """

        # Initialize PID variables if not already set
        if not hasattr(self, "prev_error"):
            self.prev_error = 0
            self.integral = 0

        # Calculate current error
        current_error = (center - 320) / 320.0  # Normalize error to [-1, 1] range

        if abs(current_error) >= 1.0:
            self.integral = 0

        self.integral += current_error
        self.integral = max(min(self.integral, 200), -200)

        derivative = current_error - self.prev_error

        self.prev_error = current_error
        pid_output = self.Kp * current_error + self.Ki * self.integral + self.Kd * derivative

        pid_output = max(min(pid_output, 7.5), -7.5)

        """
        # Calculate integral term with anti-windup
        self.integral += current_error
        if self.integral > 800:  # Limit integral windup
            self.integral = 800
        elif self.integral < -800:
            self.integral = -800

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
        """

        # Adjust velocity based on curve sharpness (slow down in curves)
        curve_factor = abs(pid_output) / 7.5  # Normalized curve sharpness
        v = self.base_speed * (1.0 - 0.4 * curve_factor)  # Reduce speed in curves

        twist = Twist2DStamped(v=v, omega=-pid_output)
        # rospy.logwarn(f"moving {v} with omega {-pid_output} at error {current_error}")
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
