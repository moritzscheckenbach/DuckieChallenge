#!/usr/bin/env python3

import os
from enum import Enum

import cv2
import numpy as np
import rospkg
import rospy
import yaml
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from normal_lane_following.msg import MultiMaskGroups
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Float64, Int32MultiArray, String
from ultralytics import YOLO


class NormalLaneFollowingLeft(DTROS):
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(NormalLaneFollowingLeft, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        self.cv_image = None  # Placeholder for the current image
        self._vehicle_name = os.environ["VEHICLE_NAME"]

        # Node activation via control mode
        self.node_active = False
        self._mode_topic = f"/{self._vehicle_name}/current_mode"
        self.sub_modus = rospy.Subscriber(self._mode_topic, Int32MultiArray, self.activate_node, queue_size=1)

        # Load configuration parameters
        self.config = self._load_config()

        # Swap defaults: for left-lane following white is left boundary and yellow is right boundary
        orig_white = self.config["defaults"]["center_white"]
        orig_yellow = self.config["defaults"]["center_yellow"]
        self.default_center_white = orig_yellow
        self.default_center_yellow = orig_white
        self.default_center_dotted = self.config["defaults"]["center_dotted"]

        # Processing parameters
        self.Xth_frame = self.config["processing"]["use_every_Xth_frame"]
        self.crop_height_percentage = self.config["processing"]["crop_height_percentage"]
        self.roi_height = self.config["processing"]["roi_height"]

        # Red stop parameters
        self.red_stop_roi_window_height = self.config["red_stop"]["window_height"]
        self.red_stop_roi_window_width = self.config["red_stop"]["window_width"]
        self.red_stop_roi_window_bottom_offset = self.config["red_stop"]["window_bottom_offset"]

        # Camera topic
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"

        # Publishers and subscribers (activated when node_active)
        if self.node_active:
            self.pub_lane = rospy.Publisher(f"/{self._vehicle_name}/detect/lane_left", Float64, queue_size=1)
            self.sub = rospy.Subscriber(f"/{self._vehicle_name}/detect/masks", MultiMaskGroups, self.callback, queue_size=1)
            self.counter = 0
            self.bridge = CvBridge()
            self.sub_image_original = rospy.Subscriber(self._camera_topic, CompressedImage, self.load_image, queue_size=1)

    def activate_node(self, msg):
        # Activate/deactivate based on control mode (True means left-lane mode)
        if msg.data[2] == True:
            rospy.loginfo("Left-Lane Following Node is active.")
            self.node_active = True
        else:
            rospy.loginfo("Left-Lane Following Node is inactive.")
            self.node_active = False

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

    def load_image(self, image_msg):
        if self.counter % self.Xth_frame != 0:
            self.counter += 1
            return
        self.counter += 1
        np_arr = np.frombuffer(image_msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.cv_image = self.crop_img(cv_image)

    def callback(self, msg: MultiMaskGroups):
        if not self.node_active:
            return
        # Convert masks
        white_masks = [self.bridge.imgmsg_to_cv2(m, desired_encoding="mono8") for m in msg.white]
        yellow_masks = [self.bridge.imgmsg_to_cv2(m, desired_encoding="mono8") for m in msg.yellow]
        # Use same logic but swapped defaults
        self.find_lane(white_masks, yellow_masks)

    def crop_img(self, img):
        img = img.copy()
        h, w = img.shape[:2]
        crop_height = int(h * self.crop_height_percentage)
        img = img[crop_height:, :]
        self.image_height = img.shape[0]
        rospy.loginfo(f"image size: height:{img.shape[0]}, width:{img.shape[1]}")
        return img

    def extract_lane_center_from_mask(self, mask, height_roi):
        if mask is None or mask.size == 0:
            return None
        row_indices = np.where(mask[height_roi, :] > 0)[0]
        return np.mean(row_indices) if len(row_indices) > 0 else None

    def find_lane(self, white_masks, yellow_masks):
        # white_masks = left boundary, yellow_masks = right boundary
        white_lane_mask = white_masks[0] if white_masks else None
        yellow_lane_mask = yellow_masks[0] if yellow_masks else None

        default_offset = (self.default_center_white - self.default_center_yellow) / 2 - 100
        c_white = self.extract_lane_center_from_mask(white_lane_mask, self.roi_height) or self.default_center_white
        c_yellow = self.extract_lane_center_from_mask(yellow_lane_mask, self.roi_height) or self.default_center_yellow

        found_white = c_white is not None
        found_yellow = c_yellow is not None

        if found_white and found_yellow:
            lane_center = (c_white + c_yellow) / 2
        elif found_white:
            lane_center = c_white - default_offset
        elif found_yellow:
            lane_center = c_yellow + default_offset
        else:
            rospy.logwarn("No lane masks detected. Using defaults.")
            lane_center = (self.default_center_white + self.default_center_yellow) / 2

        msg = Float64()
        msg.data = float(lane_center)
        self.pub_lane.publish(msg)
        self.visualize_lane(self.cv_image, lane_center, c_white, c_yellow)

    def visualize_lane(self, img, lane_center, center_white=None, center_yellow=None):
        h, w = img.shape[:2]
        vis_image = img.copy()
        # draw ROI line
        cv2.line(vis_image, (0, self.roi_height), (w, self.roi_height), (255, 0, 0), 1)
        # center of image
        cv2.line(vis_image, (int(w / 2), 0), (int(w / 2), h), (255, 255, 0), 2)
        # left (white) boundary
        if center_white is not None:
            cv2.circle(vis_image, (int(center_white), self.roi_height), 8, (255, 255, 255), -1)
        # right (yellow) boundary
        if center_yellow is not None:
            cv2.circle(vis_image, (int(center_yellow), self.roi_height), 8, (0, 255, 255), -1)
        # lane center
        cv2.circle(vis_image, (int(lane_center), self.roi_height), 10, (0, 255, 0), -1)
        cv2.imshow("Left Lane Detection Visualization", vis_image)
        cv2.waitKey(1)


if __name__ == "__main__":
    rospy.init_node("NormalLaneFollowingLeft", anonymous=False)
    node = NormalLaneFollowingLeft(node_name="NormalLaneFollowingLeft")
    rospy.spin()
