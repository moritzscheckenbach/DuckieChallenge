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


class CheckRedStop(DTROS):
    def __init__(self, node_name):
        super(CheckRedStop, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)

        self._vehicle_name = os.environ["VEHICLE_NAME"]

        # Node activation via control mode
        self._node_active = False
        self._mode_topic = f"/{self._vehicle_name}/current_mode"
        self.sub_modus = rospy.Subscriber(self._mode_topic, Int32MultiArray, self.activate_node, queue_size=1)

        # Image processing necessities
        self.bridge = CvBridge()

        # Subscribe to the mask topic
        self.sub_masks = rospy.Subscriber(f"/{self._vehicle_name}/detect/masks", MultiMaskGroups, self.check_red_stop, queue_size=1)

        # Publisher for red stop detection
        self.pub_red_stop = rospy.Publisher(f"/{self._vehicle_name}/redstop_detected", Bool, queue_size=1)

        # Configuration for ROI
        self.config = self._load_config()

        rospy.loginfo(f"{self._vehicle_name}: RedStopDetectionNode initialized")

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

    def activate_node(self, msg):
        """Activate or deactivate node based on control mode"""
        # Check control mode (using normal lane following mode for example)
        if msg.data[5] == 1:  # Check if normal lane following is active
            if not self._node_active:
                rospy.loginfo(f"{self._vehicle_name}: RedStopDetectionNode activated")
            self._node_active = True
        else:
            if self._node_active:
                rospy.loginfo(f"{self._vehicle_name}: RedStopDetectionNode deactivated")
            self._node_active = False

    def check_red_stop(self, msg):
        """Process incoming masks and check for red stop line overlap with ROI"""
        if not self._node_active:
            return

        # Get all red masks from the message
        red_masks = [self.bridge.imgmsg_to_cv2(m, desired_encoding="mono8") for m in msg.red]

        if not red_masks:
            # No red masks detected
            self.pub_red_stop.publish(Bool(data=False))
            return

        # Create ROI mask
        roi_mask = np.zeros_like(red_masks[0]) if red_masks else None
        if roi_mask is not None:
            roi_mask[self.roi["y_min"] : self.roi["y_max"], self.roi["x_min"] : self.roi["x_max"]] = 255

            # Check overlap for each red mask
            for red_mask in red_masks:
                # Resize mask if needed
                if red_mask.shape != roi_mask.shape:
                    red_mask = cv2.resize(red_mask, (roi_mask.shape[1], roi_mask.shape[0]))

                # Calculate overlap
                overlap = self._calculate_mask_overlap(red_mask, roi_mask)

                # If overlap is more than 50%, publish detection
                if overlap > 50:
                    rospy.loginfo(f"{self._vehicle_name}: Red stop detected with {overlap:.2f}% overlap")
                    self.pub_red_stop.publish(Bool(data=True))
                    return

                # Create ROI mask based on configuration
                image_height = red_masks[0].shape[0] if red_masks else 480
                image_width = red_masks[0].shape[1] if red_masks else 640

                # Get settings from config
                window_height = self.config["red_stop"]["window_height"]
                window_width = self.config["red_stop"]["window_width"]
                bottom_offset = self.config["red_stop"]["window_bottom_offset"]
                detection_threshold = self.config["red_stop"]["detection_threshold"]

                # Calculate ROI coordinates
                y_max = image_height - bottom_offset
                y_min = y_max - window_height
                x_min = max(0, (image_width - window_width) // 2)
                x_max = min(image_width, x_min + window_width)

                # Create ROI mask
                roi_mask = np.zeros_like(red_masks[0]) if red_masks else None
                if roi_mask is not None:
                    roi_mask[y_min:y_max, x_min:x_max] = 255

                    # Check overlap for each red mask
                    for red_mask in red_masks:
                        # Resize mask if needed
                        if red_mask.shape != roi_mask.shape:
                            red_mask = cv2.resize(red_mask, (roi_mask.shape[1], roi_mask.shape[0]))

                        # Calculate overlap
                        overlap = self._calculate_mask_overlap(red_mask, roi_mask)

                        # If overlap is more than threshold, publish detection
                        if overlap > detection_threshold:
                            rospy.loginfo(f"{self._vehicle_name}: Red stop detected with {overlap:.2f}% overlap")
                            self.pub_red_stop.publish(Bool(data=True))
                            return

        # No sufficient overlap found
        self.pub_red_stop.publish(Bool(data=False))

    def _calculate_mask_overlap(self, detected_mask, template_mask):
        """Calculate the percentage of overlap between two masks"""
        if detected_mask is None or template_mask is None:
            return 0

        # Calculate intersection area
        intersection = cv2.bitwise_and(detected_mask, template_mask)
        intersection_area = np.count_nonzero(intersection)

        # Calculate template area
        template_area = np.count_nonzero(template_mask)

        # Calculate percentage overlap
        if template_area == 0:
            return 0
        overlap_percentage = (intersection_area / template_area) * 100

        return overlap_percentage


if __name__ == "__main__":
    node = CheckRedStop(node_name="red_stop_detection_node")
    rospy.spin()
