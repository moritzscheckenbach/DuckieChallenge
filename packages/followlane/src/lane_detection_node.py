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
from std_msgs.msg import Float64
from ultralytics import YOLO


class DetectLaneNode(DTROS):
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(DetectLaneNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        self._vehicle_name = os.environ["VEHICLE_NAME"]

        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"

        self.pub_lane = rospy.Publisher(f"/{self._vehicle_name}/detect/lane", Float64, queue_size=1)

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

        # Image processing nessecities
        self.counter = 0
        self.bridge = CvBridge()

        self.sub_image_original = rospy.Subscriber(self._camera_topic, CompressedImage, self.cbFindLane, queue_size=1)

    def crop_img(self, img):
        img = img.copy()
        h, w = img.shape[:2]
        crop_height = int(h * 0.375)  # Crop 37.5% from the top
        img = img[crop_height:, :]
        return img

        # NOTE: If needed place bird's eye view transformation here

    # NOTE: YOLO Mask Processing (optional)
    # def process_segmentation_mask(self, mask, original_size):
    #     """Process a segmentation mask to fit the original image size."""

    def extract_lane_center_from_mask(self, mask, height_roi=325):
        if mask is None or mask.size == 0:
            rospy.logwarn("Empty mask provided for lane center extraction.")
            return None

        # Get points at the specified height
        row_indices = np.where(mask[height_roi, :] > 0)[0]
        if len(row_indices) > 0:
            return np.mean(row_indices)
        return None

    def process_segmentation_mask(self, mask, original_size):
        """Process a segmentation mask to fit the original image size."""
        if mask is None:
            return None

        # Make sure mask is properly shaped and not empty
        if mask.size == 0 or len(mask.shape) < 2:
            rospy.logwarn(f"Invalid mask shape: {mask.shape}")
            return None

        # Ensure mask is a proper numpy array with correct dimensionality
        mask = mask.squeeze()  # Remove singleton dimensions if any

        # Convert to binary mask if needed (in case it's a probability map)
        if mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8)

        # Resize mask to original image dimensions if needed
        try:
            if mask.shape[:2] != original_size[:2]:
                # Make sure both dimensions are non-zero
                if mask.shape[0] > 0 and mask.shape[1] > 0:
                    # Convert to uint8 before resizing
                    mask_uint8 = mask.astype(np.uint8)
                    resized_mask = cv2.resize(mask_uint8, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
                    return resized_mask
                else:
                    rospy.logwarn(f"Invalid mask dimensions for resizing: {mask.shape}")
                    return None
            return mask
        except Exception as e:
            rospy.logwarn(f"Error processing mask: {e}")
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

        # Apply YOLO model for lane segmentation
        try:
            results = self._model(cv_image)

            default_center_white = 600  # Default value for fallback
            default_center_yellow = 100  # Default value for fallback
            default_lane_center_from_outer_line = (default_center_white - default_center_yellow) / 2

            white_lane_mask = None
            yellow_lane_mask = None

            center_white = default_center_white
            center_yellow = default_center_yellow

            rospy.logwarn(f"Yolo init")

            # Behavior selection based on detected classes
            if results is not None and hasattr(results[0], "masks") and results[0].masks is not None:
                # Get all detected classes
                detected_classes = results[0].boxes.cls.cpu().numpy().astype(int)
                rospy.logwarn(f"1")
                rospy.logwarn(f"Detected classes: {detected_classes}")

                ROIW = False
                ROIY = False

                if 1 in detected_classes:

                    # Extract white lane class
                    white_indices = [i for i, cls in enumerate(results[0].boxes.cls) if int(cls) == 1]
                    raw_white_lane_mask = results[0].masks[white_indices[0]].data.cpu().numpy()
                    white_lane_mask = self.process_segmentation_mask(raw_white_lane_mask, cv_image.shape)
                    center_white = self.extract_lane_center_from_mask(white_lane_mask)
                    if center_white is None:
                        ROIW = False
                    else:
                        ROIW = True

                if 2 in detected_classes:

                    # Extract yellow lane class
                    yellow_indices = [i for i, cls in enumerate(results[0].boxes.cls) if int(cls) == 2]
                    raw_yellow_lane_mask = results[0].masks[yellow_indices[0]].data.cpu().numpy()
                    yellow_lane_mask = self.process_segmentation_mask(raw_yellow_lane_mask, cv_image.shape)
                    center_yellow = self.extract_lane_center_from_mask(yellow_lane_mask)
                    if center_yellow is None:
                        ROIY = False
                    else:
                        ROIY = True

                # Check if we have at least one detection of class 0 (white line) and class 1 (yellow line)
                if ROIW == True and ROIY == True:

                    lane_center = (center_white + center_yellow) / 2

                elif ROIW == True and ROIY == False:

                    lane_center = center_white - default_lane_center_from_outer_line

                elif ROIY == True and ROIW == False:

                    lane_center = center_yellow + default_lane_center_from_outer_line

                elif ROIW == False and ROIY == False:
                    rospy.logwarn(f"5")
                    rospy.logwarn("No lane masks detected by YOLO. Using default values.")
                    center_white = default_center_white
                    center_yellow = default_center_yellow
                    lane_center = (center_white + center_yellow) / 2

            else:
                rospy.logwarn(f"6")
                rospy.logwarn("No lane masks detected by YOLO. Using default values.")
                center_white = default_center_white
                center_yellow = default_center_yellow
                lane_center = (center_white + center_yellow) / 2

            # Visualize the results
            self.visualize_lane(img_orig, lane_center, white_lane_mask, yellow_lane_mask, center_white, center_yellow)

        except Exception as e:
            rospy.logwarn(f"YOLO processing error: {e}. Using default values.")

    def visualize_lane(self, img_orig, lane_center, white_lane_mask=None, yellow_lane_mask=None, center_white=None, center_yellow=None):
        """
        Create a visualization with three panels:
        1. Original image
        2. Original image with segmentation overlays
        3. Original image with center points
        """
        h, w = img_orig.shape[:2]

        # Create a blank canvas for three images side by side
        full_width = w * 3
        vis_image = np.zeros((h, full_width, 3), dtype=np.uint8)

        # Panel 1: Original image
        vis_image[:, 0:w] = img_orig

        # Panel 2: Original with segmentation overlays
        overlay = img_orig.copy()
        if white_lane_mask is not None and white_lane_mask.shape[:2] == img_orig.shape[:2]:
            # Add white lane overlay in light blue
            white_overlay = np.zeros_like(img_orig)
            white_overlay[white_lane_mask > 0] = [255, 200, 0]  # Light blue color
            overlay = cv2.addWeighted(overlay, 0.7, white_overlay, 0.3, 0)

        if yellow_lane_mask is not None and yellow_lane_mask.shape[:2] == img_orig.shape[:2]:
            # Add yellow lane overlay in green
            yellow_overlay = np.zeros_like(img_orig)
            yellow_overlay[yellow_lane_mask > 0] = [0, 255, 0]  # Green color
            overlay = cv2.addWeighted(overlay, 0.7, yellow_overlay, 0.3, 0)

        vis_image[:, w : 2 * w] = overlay

        # Panel 3: Original with center points
        center_vis = img_orig.copy()

        # Draw lane center
        if lane_center is not None:
            cv2.circle(center_vis, (int(lane_center), h - 50), 10, (0, 0, 255), -1)  # Red circle
            cv2.line(center_vis, (int(lane_center), 0), (int(lane_center), h), (0, 0, 255), 2)

        # Draw white lane center if available
        if center_white is not None:
            cv2.circle(center_vis, (int(center_white), h - 50), 8, (255, 200, 0), -1)  # Light blue circle
            cv2.line(center_vis, (int(center_white), 0), (int(center_white), h), (255, 200, 0), 2)

        # Draw yellow lane center if available
        if center_yellow is not None:
            cv2.circle(center_vis, (int(center_yellow), h - 50), 8, (0, 255, 0), -1)  # Green circle
            cv2.line(center_vis, (int(center_yellow), 0), (int(center_yellow), h), (0, 255, 0), 2)

        vis_image[:, 2 * w : 3 * w] = center_vis

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis_image, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(vis_image, "Segmentation", (w + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(vis_image, "Lane Centers", (2 * w + 10, 30), font, 1, (255, 255, 255), 2)

        # Display the visualization
        cv2.imshow("Lane Detection Visualization", vis_image)
        cv2.waitKey(1)


if __name__ == "__main__":

    node = DetectLaneNode(node_name="detect_lane_node")
    rospy.spin()
