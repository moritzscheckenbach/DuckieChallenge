#!/usr/bin/env python3


import math
import os
import time

import rospkg
import rospy
import yaml
from default.msg import BoundingBox, BoundingBoxArray
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from std_msgs.msg import Bool, Float64, Int32, Int32MultiArray, String


class ParkingManager(DTROS):
    def __init__(self, node_name):
        super(ParkingManager, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        # rospy.logwarn("Hallo, ich bin der DetectionCheckerNode!")
        self.config = self._load_config()

        self._vehicle_name = os.environ["VEHICLE_NAME"]

        self._node_active = False

        self.sub_control_mode = rospy.Subscriber(f"/{self._vehicle_name}/current_mode", Int32MultiArray, self.cbControlMode, queue_size=1)

        self.pub_cmd_vel = rospy.Publisher(f"/{self._vehicle_name}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)

        self.pub_in_region = rospy.Publisher(f"/{self._vehicle_name}/duckiebot_parked", Bool, queue_size=1)

        rospy.loginfo(f"[DetectionCheckerNode] Läuft.")

    def _load_config(self):
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("default")  # Name deines Packages!
        config_path = os.path.join(package_path, "config", "processing_params.yaml")

        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                    rospy.loginfo(f"Loaded configuration from {config_path}")
                    return config
            else:
                rospy.logerr(f"Config file not found: {config_path}")
                return None
        except Exception as e:
            rospy.logerr(f"Error loading config file: {e}.")
            return None

    def cbControlMode(self, msg: Int32MultiArray):
        if msg.data[11] == 1:
            self._node_active = True
            rospy.logwarn(f"{self._vehicle_name}: Parking is now active")
            self.park()
        else:
            self._node_active = False

    def park(self):

        # Create message for velocity commands
        cmd_msg = Twist2DStamped()

        start_time = time.time()
        rate = rospy.Rate(10)  # 10Hz control loop
        back_time = 1.5

        rospy.loginfo("Parking")

        # drive backwards
        backwards_time = time.time()
        while time.time() - backwards_time < back_time:
            cmd_msg.v = -0.35
            cmd_msg.omega = 0.0
            self.pub_cmd_vel.publish(cmd_msg)
            rate.sleep()

        cmd_msg.v = 0.0
        cmd_msg.omega = 0.0
        self.pub_cmd_vel.publish(cmd_msg)
        rospy.sleep(0.5)

        # turn left
        omega = 7  # rad/s nach links (negativ)
        duration = 1.5  # math.pi / (2 * abs(omega))  # Zeit für 90° Drehung
        rate = rospy.Rate(10)
        start_time = time.time()

        cmd_msg.v = 0.35
        cmd_msg.omega = omega

        while time.time() - start_time < duration:
            self.pub_cmd_vel.publish(cmd_msg)
            rate.sleep()

        cmd_msg.v = 0.0
        cmd_msg.omega = 0.0
        self.pub_cmd_vel.publish(cmd_msg)
        rospy.sleep(0.5)

        # drive backwards
        backwards_time = time.time()
        while time.time() - backwards_time < back_time:
            cmd_msg.v = -0.35
            cmd_msg.omega = 0.0
            self.pub_cmd_vel.publish(cmd_msg)
            rate.sleep()

        cmd_msg.v = 0.0
        cmd_msg.omega = 0.0
        self.pub_cmd_vel.publish(cmd_msg)
        rospy.sleep(0.5)

        # turn right
        omega = -7  # rad/s nach links (negativ)
        duration = 1.5  # math.pi / (2 * abs(omega))  # Zeit für 90° Drehung
        rate = rospy.Rate(10)
        start_time = time.time()

        cmd_msg.v = -0.35
        cmd_msg.omega = omega

        while time.time() - start_time < duration:
            self.pub_cmd_vel.publish(cmd_msg)
            rate.sleep()

        # Stop the robot
        cmd_msg.v = 0.0
        cmd_msg.omega = 0.0
        self.pub_cmd_vel.publish(cmd_msg)

        # warten
        rospy.sleep(5.0)

        # turn left
        omega = 7  # rad/s nach links (negativ)
        duration = 1.5  # math.pi / (2 * abs(omega))  # Zeit für 90° Drehung
        rate = rospy.Rate(10)
        start_time = time.time()

        cmd_msg.v = 0.35
        cmd_msg.omega = omega

        while time.time() - start_time < duration:
            self.pub_cmd_vel.publish(cmd_msg)
            rate.sleep()

        cmd_msg.v = 0.0
        cmd_msg.omega = 0.0
        self.pub_cmd_vel.publish(cmd_msg)
        rospy.sleep(0.5)

        # drive forwards
        backwards_time = time.time()
        while time.time() - backwards_time < back_time:
            cmd_msg.v = 0.35
            cmd_msg.omega = 0.0
            self.pub_cmd_vel.publish(cmd_msg)
            rate.sleep()

        cmd_msg.v = 0.0
        cmd_msg.omega = 0.0
        self.pub_cmd_vel.publish(cmd_msg)
        rospy.sleep(0.5)

        # turn right
        omega = -7  # rad/s nach links (negativ)
        duration = 1.5  # math.pi / (2 * abs(omega))  # Zeit für 90° Drehung
        rate = rospy.Rate(10)
        start_time = time.time()

        cmd_msg.v = -0.35
        cmd_msg.omega = omega

        while time.time() - start_time < duration:
            self.pub_cmd_vel.publish(cmd_msg)
            rate.sleep()

        # Stop the robot
        cmd_msg.v = 0.0
        cmd_msg.omega = 0.0
        self.pub_cmd_vel.publish(cmd_msg)

        # Mark Parking as completed
        self.pub_in_region.publish(Bool(data=True))


if __name__ == "__main__":
    node = ParkingManager(node_name="ParkingManager")
    rospy.spin()
