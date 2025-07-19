#!/usr/bin/env python3

import os

import cv2
import numpy as np
import rospkg
import rospy
import yaml
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from normal_lane_following.msg import MultiMaskGroups
from std_msgs.msg import Bool, Int32MultiArray


class RedStopDetectionNode(DTROS):
    def __init__(self, node_name):
        super(RedStopDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle_name = os.environ["VEHICLE_NAME"]

        # Node activation via control mode
        self._node_active = False
        self._mode_topic = f"/{self._vehicle_name}/current_mode"
        self.sub_modus = rospy.Subscriber(self._mode_topic, Int32MultiArray, self.activate_node, queue_size=1)

        # Image processing necessities
        self.bridge = CvBridge()

        # Subscribe to the mask topic
        self.sub_masks = rospy.Subscriber(f"/{self._vehicle_name}/detect/masks", MultiMaskGroups, self.check_red_stop, queue_size=1)

        # Publisher for red mask detection
        self.pub_red_mask = rospy.Publisher(f"/{self._vehicle_name}/redmask_detected", Bool, queue_size=1)

        # Configuration for ROI
        self.config = self._load_config()

    def _load_config(self):
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("default")
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

    def activate_node(self, msg):
        """Activate or deactivate node based on control mode"""
        if msg.data[12] == 1:
            if not self._node_active:
                rospy.logwarn(f"{self._vehicle_name}: RedStopDetectionNode activated")
            self._node_active = True
        else:
            self._node_active = False

    def check_red_stop(self, msg):
        """Process incoming masks and check for red stop line overlap with ROI"""
        if not self._node_active:
            return

        red_masks = [self.bridge.imgmsg_to_cv2(m, desired_encoding="mono8") for m in msg.red]

        if not red_masks:
            self.pub_red_mask.publish(Bool(data=False))
            rospy.loginfo(f"{self._vehicle_name}: No red masks received")
            return

        image_height = red_masks[0].shape[0]
        image_width = red_masks[0].shape[1]

        window_height = self.config["intersection_proximity"]["window_height"]
        window_width = self.config["intersection_proximity"]["window_width"]
        bottom_offset = self.config["intersection_proximity"]["window_bottom_offset"]

        y_max = image_height - bottom_offset
        y_min = max(0, y_max - window_height)
        x_min = max(0, (image_width - window_width) // 2)
        x_max = min(image_width, x_min + window_width)

        roi_mask = np.zeros_like(red_masks[0])
        roi_mask[y_min:y_max, x_min:x_max] = 255

        for red_mask in red_masks:
            if red_mask.shape != roi_mask.shape:
                red_mask = cv2.resize(red_mask, (roi_mask.shape[1], roi_mask.shape[0]))

            overlap = self._calculate_mask_overlap(red_mask, roi_mask)

            if overlap > 0:
                rospy.loginfo(f"{self._vehicle_name}: Red stop detected with {overlap:.2f}% overlap")
                self.pub_red_mask.publish(Bool(data=True))
                return

        self.pub_red_mask.publish(Bool(data=False))

    def _calculate_mask_overlap(self, detected_mask, template_mask):
        """Calculate the percentage of overlap between two masks"""
        if detected_mask is None or template_mask is None:
            return 0

        intersection = cv2.bitwise_and(detected_mask, template_mask)
        intersection_area = np.count_nonzero(intersection)
        template_area = np.count_nonzero(template_mask)

        if template_area == 0:
            return 0

        return (intersection_area / template_area) * 100


if __name__ == "__main__":
    node = RedStopDetectionNode(node_name="red_stop_detection_node_2")
    rospy.spin()
