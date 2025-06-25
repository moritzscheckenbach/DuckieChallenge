#!/usr/bin/env python3

import os
import time

import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from sensor_msgs.msg import CompressedImage
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
        # Publishers
        # Publisher for velocity commands
        self.pub_cmd_vel = rospy.Publisher(f"/{self._vehicle_name}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)
        # Publisher for vehicle stopped signal
        self.pub_vehicle_stopped = rospy.Publisher(f"/{self._vehicle_name}/vehicle_stopped", Bool, queue_size=1)
        self.intersection_image = rospy.Publisher(f"/{self._vehicle_name}/intersection/img", CompressedImage, queue_size=1)

        # Timer for checking stop state
        self.timer = rospy.Timer(rospy.Duration(0.1), self.check_stop_state)

        # Stop wait duration (seconds)
        self.stop_wait_duration = 2.0

        rospy.loginfo(f"{self._vehicle_name}: StoppingAtIntersectionNode initialized")

        rospy.on_shutdown(self.on_shutdown)

    def activate_node(self, msg):
        """Activate or deactivate based on control mode"""
        # Check if we're in stopping at intersection mode (index 6)
        if msg.data[6] == 1:
            if not self._node_active:
                rospy.logwarn(f"{self._vehicle_name}: StoppingAtIntersectionNode activated")
            self._node_active = True
            self.stop_at_red_stop(msg)
        else:
            if self._node_active:
                rospy.loginfo(f"{self._vehicle_name}: StoppingAtIntersectionNode deactivated")
                # Reset state if we're deactivated
                self._stopping_in_progress = False
                self._stop_start_time = None
            self._node_active = False

    def stop_at_red_stop(self, msg: Bool):
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
            self.pub_vehicle_stopped.publish(Bool(data=True))

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
        self.subscribe_to_single_image()

    def subscribe_to_single_image(self):
        """Subscribe to a single camera image and then unsubscribe"""
        self.image_received = False
        self.single_image_sub = rospy.Subscriber(f"/{self._vehicle_name}/camera_node/image/compressed", CompressedImage, self.single_image_callback, queue_size=1)
        rospy.loginfo(f"{self._vehicle_name}: Waiting for a single camera image...")

    def single_image_callback(self, img_msg):
        """Process a single camera image and then unsubscribe"""
        if not self.image_received:
            # Process the image here
            rospy.loginfo(f"{self._vehicle_name}: Image received")
            # Forward the image to the intersection image topic if needed
            self.intersection_image.publish(img_msg)

            # Unsubscribe after receiving one image
            self.single_image_sub.unregister()
            self.image_received = True
            rospy.loginfo(f"{self._vehicle_name}: Unsubscribed from camera feed")

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
