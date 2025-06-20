#!/usr/bin/env python3

import os
import time

import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from std_msgs.msg import Bool, Int32MultiArray


class StopAtIntersection(DTROS):
    def __init__(self, node_name):
        super(StopAtIntersection, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)

        self._vehicle_name = os.environ["VEHICLE_NAME"]

        # Node state
        self._node_active = False
        self._stopping_in_progress = False
        self._stop_start_time = None

        # Mode subscription
        self._mode_topic = f"/{self._vehicle_name}/current_mode"
        self.sub_modus = rospy.Subscriber(self._mode_topic, Int32MultiArray, self.activate_node, queue_size=1)

        # Subscribe to red stop detection
        self.sub_red_stop = rospy.Subscriber(f"/{self._vehicle_name}/redstop_detected", Bool, self.handle_red_stop, queue_size=1)

        # Publishers
        # Publisher for velocity commands
        self.pub_cmd_vel = rospy.Publisher(f"/{self._vehicle_name}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)
        # Publisher for vehicle stopped signal
        self.pub_vehicle_stopped = rospy.Publisher(f"/{self._vehicle_name}/vehicle_stopped", Bool, queue_size=1)
        # Publisher for handling completed signal
        self.pub_stop_handled = rospy.Publisher(f"/{self._vehicle_name}/intersection_handled", Bool, queue_size=1)

        # Timer for checking stop state
        self.timer = rospy.Timer(rospy.Duration(0.1), self.check_stop_state)

        # Stop wait duration (seconds)
        self.stop_wait_duration = 2.0

        rospy.loginfo(f"{self._vehicle_name}: RedStopHandlingNode initialized")

        rospy.on_shutdown(self.on_shutdown)

    def activate_node(self, msg):
        """Activate or deactivate based on control mode"""
        # Check if we're in stopping at intersection mode (index 6)
        if msg.data[6] == 1:
            if not self._node_active:
                rospy.loginfo(f"{self._vehicle_name}: RedStopHandlingNode activated")
            self._node_active = True
        else:
            if self._node_active:
                rospy.loginfo(f"{self._vehicle_name}: RedStopHandlingNode deactivated")
                # Reset state if we're deactivated
                self._stopping_in_progress = False
                self._stop_start_time = None
            self._node_active = False

    def handle_red_stop(self, msg: Bool):
        """Handle red stop detection"""
        if not self._node_active:
            return

        if msg.data and not self._stopping_in_progress:
            # Start stopping procedure
            rospy.loginfo(f"{self._vehicle_name}: Red stop line detected, stopping vehicle")
            self._stopping_in_progress = True
            self._stop_start_time = rospy.get_time()

            # Send stop command
            self.stop_vehicle()

            # Signal that vehicle is stopped
            self.pub_vehicle_stopped.publish(Bool(data=True))

    def check_stop_state(self, event):
        """Check if we've waited long enough at the stop"""
        if not self._node_active or not self._stopping_in_progress or self._stop_start_time is None:
            return

        # Check if we've waited long enough
        current_time = rospy.get_time()
        elapsed = current_time - self._stop_start_time

        if elapsed >= self.stop_wait_duration:
            rospy.loginfo(f"{self._vehicle_name}: Stop wait completed after {elapsed:.2f} seconds")

            # Signal that intersection handling is done
            self.pub_stop_handled.publish(Bool(data=True))

            # Reset state
            self._stopping_in_progress = False
            self._stop_start_time = None

    def stop_vehicle(self):
        """Send command to stop the vehicle"""
        twist = Twist2DStamped()
        twist.v = 0.0  # Zero velocity
        twist.omega = 0.0  # Zero angular velocity
        self.pub_cmd_vel.publish(twist)
        rospy.loginfo(f"{self._vehicle_name}: Stop command sent")

    def on_shutdown(self):
        """Handle shutdown"""
        # Stop the vehicle when shutting down
        twist = Twist2DStamped()
        twist.v = 0.0
        twist.omega = 0.0
        self.pub_cmd_vel.publish(twist)
        rospy.loginfo(f"{self._vehicle_name}: Stopped")


if __name__ == "__main__":
    node = StopAtIntersection(node_name="red_stop_handling_node")
    rospy.spin()
