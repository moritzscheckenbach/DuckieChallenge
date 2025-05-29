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
from std_msgs.msg import Float64MultiArray
from ultralytics import YOLO


class DetectLaneNode(DTROS):
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(DetectLaneNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        self.load_conf("packages/followlane/config/detect_lane.yaml")
        self._vehicle_name = os.environ["VEHICLE_NAME"]
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"

        # Initialize YOLO model for lane segmentation
        yolo_model_path = "packages/followlane/src/model/yolo_v11_seg_20250528.pt"  # Path to your lane segmentation model
        # Check if the model file exists, otherwise show a warning
        if os.path.exists(yolo_model_path):
            self._model = YOLO(yolo_model_path)
            self.yolo_enabled = True
        else:
            rospy.logwarn(f"YOLO model not found at {yolo_model_path}. Running in fallback mode.")
            self._model = None
            self.yolo_enabled = False

        self.sub_image_original = rospy.Subscriber(self._camera_topic, CompressedImage, self.cbFindLane, queue_size=1)

        self.pub_lane = rospy.Publisher(f"/{self._vehicle_name}/detect/lane", Float64MultiArray, queue_size=1)
        # Add a publisher for visualization of segmentation results
        self._yolo_viz_topic = f"/{self._vehicle_name}/detect/lane/segmentation"
        self.pub_segmentation = rospy.Publisher(self._yolo_viz_topic, Image, queue_size=1)

        self.counter = 0
        self.bridge = CvBridge()

    def crop_img(self, img):
        img = img.copy()
        print(img.shape)

        pts1 = np.float32(
            [
                [self.conf["lane_image"]["top_left_x"], self.conf["lane_image"]["top_left_y"]],
                [self.conf["lane_image"]["top_right_x"], self.conf["lane_image"]["top_right_y"]],
                [self.conf["lane_image"]["bottom_right_x"], self.conf["lane_image"]["bottom_right_y"]],
                [self.conf["lane_image"]["bottom_left_x"], self.conf["lane_image"]["bottom_left_y"]],
            ]
        )

        pts2 = np.float32([[0, 0], [100, 0], [0, 100], [100, 100]])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(img, M, (100, 100))

    def process_segmentation_mask(self, mask, original_size):
        """Process a segmentation mask to fit the original image size."""
        if mask is None:
            return None

        # Resize mask to original image dimensions if needed
        if mask.shape[:2] != original_size[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        return mask

    def extract_lane_center_from_mask(self, mask, height_roi=50):
        """Extract the lane center from a segmentation mask at a specific height."""
        if mask is None or mask.size == 0:
            return None

        # Get points at the specified height
        row_indices = np.where(mask[height_roi, :] > 0)[0]
        if len(row_indices) > 0:
            return np.mean(row_indices)
        return None

    def cbFindLane(self, image_msg):
        if self.counter % 5 != 0:
            self.counter += 1
            return
        else:
            self.counter += 1

        # Convert compressed image to OpenCV format
        np_arr = np.frombuffer(image_msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Keep a copy for region of interest cropping
        img_orig = cv_image.copy()

        # Default values if no detection is made
        center_white = 100
        center_yellow = 900

        # Apply YOLO model for lane segmentation
        try:
            results = self._model(cv_image)

            # Process segmentation masks from YOLO results
            if results is not None and hasattr(results[0], "masks") and results[0].masks is not None:
                # Extract masks for white and yellow lane classes
                white_lane_mask = None
                yellow_lane_mask = None

                # Extract white lane class (assuming class index 0)
                white_indices = [i for i, cls in enumerate(results[0].boxes.cls) if int(cls) == 0]
                if white_indices and len(results[0].masks) > white_indices[0]:
                    white_lane_mask = results[0].masks[white_indices[0]].data.cpu().numpy()
                    white_lane_mask = self.process_segmentation_mask(white_lane_mask, cv_image.shape)

                # Extract yellow lane class (assuming class index 1)
                yellow_indices = [i for i, cls in enumerate(results[0].boxes.cls) if int(cls) == 1]
                if yellow_indices and len(results[0].masks) > yellow_indices[0]:
                    yellow_lane_mask = results[0].masks[yellow_indices[0]].data.cpu().numpy()
                    yellow_lane_mask = self.process_segmentation_mask(yellow_lane_mask, cv_image.shape)

                # Extract lane centers from masks
                if white_lane_mask is not None:
                    center_white_from_yolo = self.extract_lane_center_from_mask(white_lane_mask)
                    if center_white_from_yolo is not None:
                        center_white = center_white_from_yolo

                if yellow_lane_mask is not None:
                    center_yellow_from_yolo = self.extract_lane_center_from_mask(yellow_lane_mask)
                    if center_yellow_from_yolo is not None:
                        center_yellow = center_yellow_from_yolo
        except Exception as e:
            rospy.logwarn(f"YOLO processing error: {e}. Using default values.")

        # Handle NaN values
        if np.isnan(center_white):
            center_white = 100

        if np.isnan(center_yellow):
            center_yellow = 900

        calculated_center = (center_white + center_yellow) / 2

        # Create a visualization image of the cropped region
        img_cropped = self.crop_img(img_orig)
        vis_img = img_cropped.copy()

        # Draw lane centers on the visualization image
        cv2.circle(vis_img, (int(center_white), 50), 5, (255, 0, 0), -1)
        cv2.circle(vis_img, (int(center_yellow), 50), 5, (0, 255, 0), -1)
        cv2.circle(vis_img, (int(calculated_center), 50), 5, (0, 0, 255), -1)

        # Create a visualization image with segmentation overlay from YOLO
        seg_vis_img = cv_image.copy()
        try:
            if results is not None and hasattr(results[0], "plot"):
                seg_vis_img = results[0].plot()  # Let YOLO plot its segmentation
        except Exception as e:
            rospy.logwarn(f"Error plotting YOLO results: {e}")

        # Convert the OpenCV image to ROS Image and publish
        try:
            viz_msg = self.bridge.cv2_to_imgmsg(seg_vis_img, "bgr8")
            self.pub_segmentation.publish(viz_msg)
        except Exception as e:
            rospy.logwarn(f"Error converting visualization image: {e}")

        # Show images for debugging (may not work in headless environments)
        try:
            cv2.imshow("Lane Detection", vis_img)
            cv2.imshow("YOLO Lane Segmentation", seg_vis_img)
            cv2.waitKey(1)
        except Exception:
            pass  # Silently ignore display errors

        # Create array message with white center, yellow center, and calculated center
        msg_centers = Float64MultiArray()
        msg_centers.data = [float(center_white), float(center_yellow), float(calculated_center)]
        self.pub_lane.publish(msg_centers)

    def load_conf(self, path):

        with open(path, "r") as f:
            text = f.read()

        self.conf = yaml.safe_load(text)

        self.hue_white_l = self.conf["white"]["hl"]
        self.hue_white_h = self.conf["white"]["hh"]
        self.saturation_white_l = self.conf["white"]["sl"]
        self.saturation_white_h = self.conf["white"]["sh"]
        self.lightness_white_l = self.conf["white"]["vl"]
        self.lightness_white_h = self.conf["white"]["vh"]

        self.hue_yellow_l = self.conf["yellow"]["hl"]
        self.hue_yellow_h = self.conf["yellow"]["hh"]
        self.saturation_yellow_l = self.conf["yellow"]["sl"]
        self.saturation_yellow_h = self.conf["yellow"]["sh"]
        self.lightness_yellow_l = self.conf["yellow"]["vl"]
        self.lightness_yellow_h = self.conf["yellow"]["vh"]

        self.hue_duck_l = self.conf["duck"]["hl"]
        self.hue_duck_h = self.conf["duck"]["hh"]
        self.saturation_duck_l = self.conf["duck"]["sl"]
        self.saturation_duck_h = self.conf["duck"]["sh"]
        self.lightness_duck_l = self.conf["duck"]["vl"]
        self.lightness_duck_h = self.conf["duck"]["vh"]


if __name__ == "__main__":

    node = DetectLaneNode(node_name="detect_lane_node")
    rospy.spin()
