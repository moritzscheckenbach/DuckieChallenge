#!/usr/bin/env python3

import os
import time

import rospkg
import rospy
import yaml
from default.msg import BoundingBox, BoundingBoxArray
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from std_msgs.msg import Bool, Float64, Int32, Int32MultiArray, String


class ReturnToLane(DTROS):
    def __init__(self, node_name):
        super(ReturnToLane, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        # rospy.logwarn("Hallo, ich bin der DetectionCheckerNode!")
        rospy.logwarn(f"[die gescheite nodeess")
        self.config = self._load_config()

        self.target_class_id = self.config["right_check_duckie"]["target_class_id"]
        self.region = self.config["right_check_duckie"]["region"]

        self._vehicle_name = os.environ["VEHICLE_NAME"]

        self._node_active = False

        self.sub = rospy.Subscriber(f"/{self._vehicle_name}/detect/bounding_boxes", BoundingBoxArray, self.detection_callback, queue_size=1)
        self.sub_control_mode = rospy.Subscriber(f"/{self._vehicle_name}/current_mode", Int32MultiArray, self.cbControlMode, queue_size=1)

        self.pub_in_region = rospy.Publisher(f"/{self._vehicle_name}/detect/afterall_in_region", Bool, queue_size=1)
        self.pub_cmd_vel = rospy.Publisher(f"/{self._vehicle_name}/cmd_vel", Twist2DStamped, queue_size=1)
        self.pub_lane_change = rospy.Publisher(f"/{self._vehicle_name}/Change/LaneRight", Bool, queue_size=1)

        rospy.logwarn(f"[die gescheite node")
        self.timer_init = 2  # seconds
        self.timer_value = self.timer_init
        # Create timer that decrements timer_value every second
        self.in_region = False

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
        rospy.logwarn(f"Received control mode message: {msg.data}")
        if msg.data[15] == 1:
            self._node_active = True
            rospy.logwarn(f"{self._vehicle_name}: DuckieCenterRightCheck is now active")
            self.drive_right()
        else:
            self._node_active = False

    def detection_callback(self, msg):
        # rospy.logwarn(f"Received detection message with {msg.boxes} boxes.")

        if self._node_active == True:
            for detection in msg.boxes:
                if detection.class_id == self.target_class_id:
                    x_min, y_min = detection.x_min, detection.y_min
                    x_max, y_max = detection.x_max, detection.y_max

                    if not (self.region["x_min"] <= x_min <= self.region["x_max"] and self.region["y_min"] <= y_min <= self.region["y_max"]) or (
                        self.region["x_min"] <= x_max <= self.region["x_max"] and self.region["y_min"] <= y_max <= self.region["y_max"]
                    ):
                        self.in_region = False
                    else:
                        self.in_region = True
                        return
                else:
                    self.in_region = False

    def drive_right(self):
        # hard coded transition to opposite lane following
        v = 0.4
        omega = -35  # rad/s nach links (negativ)
        duration = 0.6  # math.pi / (2 * abs(omega))  # Zeit für 90° Drehung
        rate = rospy.Rate(10)
        start_time = time.time()

        cmd_msg = Twist2DStamped()
        cmd_msg.v = v
        cmd_msg.omega = omega

        while time.time() - start_time < duration:
            self.pub_cmd_vel.publish(cmd_msg)
            rate.sleep()

        if self.in_region:
            self.pub_in_region.publish(True)
        else:
            self.pub_lane_change.publish(True)


if __name__ == "__main__":
    node = ReturnToLane(node_name="ReturnToLane")
    rospy.spin()
