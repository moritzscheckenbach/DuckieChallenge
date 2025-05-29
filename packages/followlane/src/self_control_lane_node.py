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

        # --- PID Controller Initialization ---
        self.Kp = rospy.get_param("~Kp", 1.0)
        self.Ki = rospy.get_param("~Ki", 0.0)
        self.Kd = rospy.get_param("~Kd", 0.1)

        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = rospy.Time.now()

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
        # --- PID Controller ---
        target_center = 500  # Bildmitte
        error = (center - target_center) / 100.0  # Normierter Fehler

        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        # Schutz gegen sehr kleine Zeitabstände
        if dt == 0:
            dt = 1e-3

        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        self.last_error = error

        omega = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        v = 0.2  # konstante Vorwärtsgeschwindigkeit

        twist = Twist2DStamped(v=v, omega=omega)
        rospy.loginfo(f"[PID] v={v:.2f}, omega={omega:.2f}, error={error:.2f}")
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
