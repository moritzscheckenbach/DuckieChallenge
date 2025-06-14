#!/usr/bin/env python3

import os
from enum import Enum

import cv2
import numpy as np
import rospy
import yaml
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Float64
from ultralytics import YOLO


class NormalLaneFollowingNode(DTROS):
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(NormalLaneFollowingNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle_name = os.environ["VEHICLE_NAME"]

        # Load configuration parameters
        self.config = self._load_config()

        self.roi_height = self.config["processing"]["roi_height"]  # Height of the ROI for lane center extraction
        self.default_center_white = self.config["defaults"]["center_white"]  # Default value for fallback
        self.default_center_yellow = self.config["defaults"]["center_yellow"]  # Default value for fallback

        self.red_stop_roi_window_height = self.config["red_stop"]["window_height"]
        self.red_stop_roi_window_width = self.config["red_stop"]["window_width"]
        self.red_stop_roi_window_bottom_offset = self.config["red_stop"]["window_bottom_offset"]

        self.pub_lane = rospy.Publisher(f"/{self._vehicle_name}/detect/lane", Float64, queue_size=1)
        self.pub_red_stop = rospy.Publisher(f"/{self._vehicle_name}/detect/red_stop", Bool, queue_size=1)

    def _load_config(self):
        """Load configuration from YAML file with fallback to default values."""
        config_path = "packages/normal_lane_following/config/normal_lane_following_params.yaml"

        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                    rospy.loginfo(f"Loaded configuration from {config_path}")
                    return config
            else:
                rospy.logerr(f"Error loading config file: {e}.")
        except Exception as e:
            rospy.logerr(f"Error loading config file: {e}.")


if __name__ == "__main__":

    node = NormalLaneFollowingNode(node_name="detect_lane_node")
    rospy.spin()
