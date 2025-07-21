#!/usr/bin/env python3

import os
import time
from enum import Enum

import cv2
import numpy as np
import rospkg
import rospy
import yaml
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from normal_lane_following.msg import MultiMaskGroups
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Float64, Int32MultiArray, String
from ultralytics import YOLO


class OppositeLaneFollowing(DTROS):
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(OppositeLaneFollowing, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        self.cv_image = None  # Placeholder for the current image

        self._vehicle_name = os.environ["VEHICLE_NAME"]

        self.image_height = 480  # Default image height before cropping

        """
        Angepasst an switch_control_node.py
        """
        self.node_active = False  # Flag to check if the node is active
        self._mode_topic = f"/{self._vehicle_name}/current_mode"
        self.sub_modus = rospy.Subscriber(self._mode_topic, Int32MultiArray, self.ActivateNode, queue_size=1)

        # Load configuration parameters
        self.config = self._load_config()

        self.Xth_frame = self.config["processing"]["use_every_Xth_frame"]  # Process every Xth frame
        self.crop_height_percentage = self.config["processing"]["crop_height_percentage"]  # Percentage of the image height to crop from the top
        self.roi_height = self.config["processing"]["roi_height"]  # Height of the ROI for lane center extraction
        self.default_center_white = self.config["defaults"]["center_white"]  # Default value for fallback
        self.default_center_yellow = self.config["defaults"]["center_yellow"]  # Default value for fallback
        self.default_center_dotted = self.config["defaults"]["center_dotted"]

        self.red_stop_roi_window_height = self.config["red_stop"]["window_height"]
        self.red_stop_roi_window_width = self.config["red_stop"]["window_width"]
        self.red_stop_roi_window_bottom_offset = self.config["red_stop"]["window_bottom_offset"]

        self.red_stop_roi_window_height = self.config["red_stop"]["window_height"]
        self.red_stop_roi_window_width = self.config["red_stop"]["window_width"]
        self.red_stop_roi_window_bottom_offset = self.config["red_stop"]["window_bottom_offset"]

        self.pub_cmd_vel = rospy.Publisher(f"/{self._vehicle_name}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)
        self.pub_lane = rospy.Publisher(f"/{self._vehicle_name}/detect/lane", Float64, queue_size=1)
        self.sub = rospy.Subscriber(f"/{self._vehicle_name}/detect/masks", MultiMaskGroups, self.callback, queue_size=1)

        # Image processing nessecities
        self.counter = 0
        self.bridge = CvBridge()

        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self.sub_image_original = rospy.Subscriber(self._camera_topic, CompressedImage, self.LoadImage, queue_size=1)

    def ActivateNode(self, msg):
        """
        Callback to activate or deactivate the node based on the current mode.
        """
        if msg.data[2] == 1:

            # hard coded transition to opposite lane following
            v = 0.35
            omega = 5  # rad/s nach links (negativ)
            duration = 1  # math.pi / (2 * abs(omega))  # Zeit für 90° Drehung
            rate = rospy.Rate(10)
            start_time = time.time()

            cmd_msg = Twist2DStamped()
            cmd_msg.v = v
            cmd_msg.omega = omega

            while time.time() - start_time < duration:
                self.pub_cmd_vel.publish(cmd_msg)
                rate.sleep()

            rospy.logwarn("Opposite Lane Following Node is active.")
            self.node_active = True
        else:
            self.node_active = False

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

    def LoadImage(self, image_msg):
        if self.counter % self.Xth_frame != 0:
            self.counter += 1
            return
        else:
            self.counter += 1

        # Convert compressed image to OpenCV format
        np_arr = np.frombuffer(image_msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.cv_image = self.crop_img(cv_image)

    def callback(self, msg: MultiMaskGroups):
        if self.node_active:
            # Wandelt sensor_msgs/Image[] in OpenCV-Bilder um
            white_masks = [self.bridge.imgmsg_to_cv2(m, desired_encoding="mono8") for m in msg.white]
            yellow_masks = [self.bridge.imgmsg_to_cv2(m, desired_encoding="mono8") for m in msg.yellow]
            red_masks = [self.bridge.imgmsg_to_cv2(m, desired_encoding="mono8") for m in msg.red]
            dotted_masks = [self.bridge.imgmsg_to_cv2(m, desired_encoding="mono8") for m in msg.dotted]

            rospy.loginfo(f"Erhalten: {len(white_masks)} weiße, {len(yellow_masks)} gelbe, {len(red_masks)} rote Masken, {len(dotted_masks)} dotted Masken")

            self.FindLane(white_masks, yellow_masks)

        # Beispiel: Zeige erste weiße Maske (falls vorhanden)
        # if white_masks:
        #     cv2.imshow("Weiße Maske 0", white_masks[0])
        #     cv2.waitKey(1)

    def crop_img(self, img):
        img = img.copy()
        h, w = img.shape[:2]
        crop_height = int(h * self.crop_height_percentage)  # Crop X% from the top
        img = img[crop_height:, :]
        self.image_height = img.shape[0]
        # rospy.loginfo(f"image size: height:{img.shape[0]}, width:{img.shape[1]}")

        return img

        # NOTE: If needed place bird's eye view transformation here

    def extract_lane_center_from_mask(self, mask, height_roi):
        if mask is None or mask.size == 0:
            rospy.logwarn("Empty mask provided for lane center extraction.")
            return None

        # Get points at the specified height
        row_indices = np.where(mask[height_roi, :] > 0)[0]
        if len(row_indices) > 0:
            return np.mean(row_indices)
        return None

    def FindLane(self, white_masks, yellow_masks):

        try:

            white_lane_mask = white_masks[0] if white_masks else None
            yellow_lane_mask = yellow_masks[0] if yellow_masks else None

            default_lane_center_from_outer_line = (self.default_center_white - self.default_center_yellow) / 2 - 100

            center_white = self.default_center_white
            center_yellow = self.default_center_yellow
            center_dotted = self.default_center_dotted

            ROIW = False
            ROIY = False

            center_white = self.extract_lane_center_from_mask(white_lane_mask, self.roi_height)
            if center_white is None:
                ROIW = False
            else:
                ROIW = True

            center_yellow = self.extract_lane_center_from_mask(yellow_lane_mask, self.roi_height)
            if center_yellow is None:
                ROIY = False
            else:
                ROIY = True

            # Check if we have at least one detection of class 1 (white line) and class 2 (yellow line)
            if ROIW == True and ROIY == True:
                if center_white < center_yellow:
                    lane_center = (center_white + center_yellow) / 2
                else:
                    lane_center = center_yellow - default_lane_center_from_outer_line

            elif ROIW == True and ROIY == False:

                lane_center = center_white + default_lane_center_from_outer_line

            elif ROIY == True and ROIW == False:

                lane_center = center_yellow - default_lane_center_from_outer_line

            elif ROIW == False and ROIY == False:
                rospy.logwarn("No lane masks detected by YOLO. Using default values.")
                center_white = self.default_center_white
                center_yellow = self.default_center_yellow
                lane_center = (center_white + center_yellow) / 2

            else:
                rospy.logwarn("No lane masks detected by YOLO. Using default values.")
                center_white = self.default_center_white
                center_yellow = self.default_center_yellow
                lane_center = (center_white + center_yellow) / 2

            lane_center_msg = Float64()
            lane_center_msg.data = float(lane_center)
            self.pub_lane.publish(lane_center_msg)

            # Visualize the results
            self.visualize_lane(
                self.cv_image,
                lane_center,
                center_white,
                center_yellow,
            )

        except Exception as e:
            rospy.logwarn(f"Lane Center Compute Error: {e}. Using default values.")

    def visualize_lane(self, img, lane_center, center_white=None, center_yellow=None):
        """
        Create a visualization with three panels:
        1. Original image
        2. Original image with segmentation overlays
        3. Original image with center points
        """
        h, w = img.shape[:2]

        full_width = w
        vis_image = np.zeros((h, full_width, 3), dtype=np.uint8)

        # Panel 2: Original with center points
        center_vis = img.copy()

        # Horizontale Linie bei ROI-Höhe
        cv2.line(center_vis, (0, self.roi_height), (w, self.roi_height), (255, 0, 0), 1)  # Blaue Linie

        # Draw image center
        cv2.line(center_vis, (int(w / 2), 0), (int(w / 2), h), (255, 255, 0), 2)

        # Draw white lane center if available
        if center_white is not None:
            cv2.circle(center_vis, (int(center_white), self.roi_height), 8, (255, 255, 255), -1)
            cv2.line(center_vis, (int(center_white), 0), (int(center_white), h), (255, 255, 255), 2)

        # Draw yellow lane center if available
        if center_yellow is not None:
            cv2.circle(center_vis, (int(center_yellow), self.roi_height), 8, (0, 255, 255), -1)
            cv2.line(center_vis, (int(center_yellow), 0), (int(center_yellow), h), (0, 255, 255), 2)

        # Draw lane center
        if lane_center is not None:
            cv2.circle(center_vis, (int(lane_center), self.roi_height), 10, (0, 255, 0), -1)
            cv2.line(center_vis, (int(lane_center), 0), (int(lane_center), h), (0, 255, 0), 2)
            cv2.line(center_vis, (int(lane_center), self.roi_height), (int(w / 2), self.roi_height), (0, 0, 255), 2)

        vis_image[:, 0:w] = center_vis

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis_image, "Left Lane PID", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(vis_image, "Lane Centers", (w + 10, 30), font, 1, (255, 255, 255), 2)

        # Display the visualization
        cv2.imshow("Lane Detection Visualization", vis_image)
        cv2.waitKey(1)


if __name__ == "__main__":

    node = OppositeLaneFollowing(node_name="OppositeLaneFollowing")
    rospy.spin()
