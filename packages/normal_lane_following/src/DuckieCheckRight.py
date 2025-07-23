#!/usr/bin/env python3

import os

import rospkg
import rospy
import yaml
from default.msg import BoundingBox, BoundingBoxArray
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import Bool, Float64, Int32, Int32MultiArray, String


class DetectionCheckerRightNode(DTROS):
    def __init__(self, node_name):
        super(DetectionCheckerRightNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        # rospy.logwarn("Hallo, ich bin der DetectionCheckerNode!")
        self.config = self._load_config()

        self.target_class_id = self.config["right_check_duckie"]["target_class_id"]
        self.region = self.config["right_check_duckie"]["region"]

        self._vehicle_name = os.environ["VEHICLE_NAME"]

        self._node_active = False

        self.sub = rospy.Subscriber(f"/{self._vehicle_name}/detect/bounding_boxes", BoundingBoxArray, self.detection_callback, queue_size=1)
        self.sub_control_mode = rospy.Subscriber(f"/{self._vehicle_name}/current_mode", Int32MultiArray, self.cbControlMode, queue_size=1)

        self.pub_in_region = rospy.Publisher(f"/{self._vehicle_name}/detect/not_in_region", Bool, queue_size=1)

        rospy.loginfo(f"[DetectionCheckerNode] Läuft. Überwacht Klasse {self.target_class_id} im Bereich {self.region}")
        self.timer_init = 3  # seconds
        self.timer_value = self.timer_init
        # Create timer that decrements timer_value every second
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)

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
        if msg.data[3] == 1:
            self._node_active = True
            rospy.logwarn(f"{self._vehicle_name}: DuckieCenterRightCheck is now active")
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
                        break
                    else:
                        self.timer_value = self.timer_init

    def timer_callback(self, event):
        if self._node_active:
            self.timer_value = self.timer_value - 0.1
            rospy.logwarn(f"Timer value: {self.timer_value}")
            if self.timer_value <= 0.0:
                rospy.logwarn("Timer expired - duckie has left the region")
                self.pub_in_region.publish(True)
                self.timer_value = self.timer_init


if __name__ == "__main__":
    node = DetectionCheckerRightNode(node_name="DetectionCheckerRightNode")
    rospy.spin()
