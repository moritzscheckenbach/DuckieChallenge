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


class DetectLaneNode(DTROS):
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(DetectLaneNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        self._vehicle_name = os.environ["VEHICLE_NAME"]
        self.image_height = 480  # Default image height before cropping
        self.roi_height = 100  # Height of the ROI for lane center extraction

        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"

        self.pub_lane = rospy.Publisher(f"/{self._vehicle_name}/detect/lane", Float64, queue_size=1)
        self.pub_red_stop = rospy.Publisher(f"/{self._vehicle_name}/detect/red_stop", Bool, queue_size=1)

        # Initialize YOLO model for lane segmentation
        yolo_model_path = "packages/lane_detection/src/model/yolo_v11_lane_seg_20250528.pt"  # Path to your lane segmentation model
        # Check if the model file exists, otherwise show a warning
        if os.path.exists(yolo_model_path):
            self._model = YOLO(yolo_model_path)
            self.yolo_enabled = True
            if self._model is not None:
                class_names = self._model.names  # Holt alle Klassennamen
                rospy.loginfo(f"YOLO Model Classes: {class_names}")

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
        self.image_height = img.shape[0]
        rospy.loginfo(f"image size: height:{img.shape[0]}, width:{img.shape[1]}")

        return img

        # NOTE: If needed place bird's eye view transformation here

    # NOTE: YOLO Mask Processing (optional)
    # def process_segmentation_mask(self, mask, original_size):
    #     """Process a segmentation mask to fit the original image size."""

    def extract_lane_center_from_mask(self, mask, height_roi):
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
        cv_image = self.crop_img(cv_image)

        # Apply YOLO model for lane segmentation
        try:
            results = self._model(cv_image)

            default_center_white = 600  # Default value for fallback
            default_center_yellow = 40  # Default value for fallback
            default_lane_center_from_outer_line = (default_center_white - default_center_yellow) / 2 - 100

            white_lane_mask = None
            yellow_lane_mask = None
            red_stop_mask = None

            all_white_masks = []
            all_yellow_masks = []
            all_red_masks = []

            center_white = default_center_white
            center_yellow = default_center_yellow

            # Behavior selection based on detected classes
            if results is not None and hasattr(results[0], "masks") and results[0].masks is not None:
                # Get all detected classes
                detected_classes = results[0].boxes.cls.cpu().numpy().astype(int)
                rospy.logwarn(f"Detected classes: {detected_classes}")

                ROIW = False
                ROIY = False

                if 1 in detected_classes:
                    # Extract white lane class
                    white_indices = [i for i, cls in enumerate(results[0].boxes.cls) if int(cls) == 1]
                    raw_white_lane_mask = results[0].masks[white_indices[0]].data.cpu().numpy()
                    white_lane_mask = self.process_segmentation_mask(raw_white_lane_mask, cv_image.shape)
                    center_white = self.extract_lane_center_from_mask(white_lane_mask, self.roi_height)
                    if center_white is None:
                        ROIW = False
                    else:
                        ROIW = True

                    # Für die Visualisierung: Sammeln aller weißen Masken
                    for idx in white_indices:
                        raw_mask = results[0].masks[idx].data.cpu().numpy()
                        mask = self.process_segmentation_mask(raw_mask, cv_image.shape)
                        if mask is not None:
                            all_white_masks.append(mask)

                if 2 in detected_classes:
                    # Extract yellow lane class
                    yellow_indices = [i for i, cls in enumerate(results[0].boxes.cls) if int(cls) == 2]
                    raw_yellow_lane_mask = results[0].masks[yellow_indices[0]].data.cpu().numpy()
                    yellow_lane_mask = self.process_segmentation_mask(raw_yellow_lane_mask, cv_image.shape)
                    center_yellow = self.extract_lane_center_from_mask(yellow_lane_mask, self.roi_height)
                    if center_yellow is None:
                        ROIY = False
                    else:
                        ROIY = True

                    # Für die Visualisierung: Sammeln aller gelben Masken
                    for idx in yellow_indices:
                        raw_mask = results[0].masks[idx].data.cpu().numpy()
                        mask = self.process_segmentation_mask(raw_mask, cv_image.shape)
                        if mask is not None:
                            all_yellow_masks.append(mask)

                if 0 in detected_classes:
                    # Extract red stop class
                    red_indices = [i for i, cls in enumerate(results[0].boxes.cls) if int(cls) == 0]
                    if red_indices:
                        raw_red_stop_mask = results[0].masks[red_indices[0]].data.cpu().numpy()
                        red_stop_mask = self.process_segmentation_mask(raw_red_stop_mask, cv_image.shape)

                    # Für die Visualisierung: Sammeln aller roten Masken
                    for idx in red_indices:
                        raw_mask = results[0].masks[idx].data.cpu().numpy()
                        mask = self.process_segmentation_mask(raw_mask, cv_image.shape)
                        if mask is not None:
                            all_red_masks.append(mask)

                    red_stop_roi_window_height = 100
                    red_stop_roi_window_width = 400
                    red_stop_roi_window_y_start = cv_image.shape[0] - 100 - red_stop_roi_window_height
                    red_stop_roi_window_x_start = (cv_image.shape[1] - red_stop_roi_window_width) // 2

                    window_mask = mask[
                        red_stop_roi_window_y_start : red_stop_roi_window_y_start + red_stop_roi_window_height, red_stop_roi_window_x_start : red_stop_roi_window_x_start + red_stop_roi_window_width
                    ]

                    if window_mask.size > 0:
                        mask_percentage = np.sum(window_mask > 0) / window_mask.size * 100

                        if mask_percentage > 50:
                            rospy.logwarn(f"Red mask detected in defined window! Coverage: {mask_percentage:.2f}%")
                            red_stop_msg = Bool()
                            red_stop_msg.data = True
                            self.pub_red_stop.publish(red_stop_msg)
                        else:
                            red_stop_msg = Bool()
                            red_stop_msg.data = False
                            self.pub_red_stop.publish(red_stop_msg)

                # Check if we have at least one detection of class 1 (white line) and class 2 (yellow line)
                if ROIW == True and ROIY == True:
                    if center_white > center_yellow:
                        lane_center = (center_white + center_yellow) / 2
                    else:
                        lane_center = center_yellow + default_lane_center_from_outer_line

                elif ROIW == True and ROIY == False:

                    lane_center = center_white - default_lane_center_from_outer_line

                elif ROIY == True and ROIW == False:

                    lane_center = center_yellow + default_lane_center_from_outer_line

                elif ROIW == False and ROIY == False:
                    rospy.logwarn("No lane masks detected by YOLO. Using default values.")
                    center_white = default_center_white
                    center_yellow = default_center_yellow
                    lane_center = (center_white + center_yellow) / 2

            else:
                rospy.logwarn("No lane masks detected by YOLO. Using default values.")
                center_white = default_center_white
                center_yellow = default_center_yellow
                lane_center = (center_white + center_yellow) / 2

            lane_center_msg = Float64()
            lane_center_msg.data = float(lane_center)
            self.pub_lane.publish(lane_center_msg)

            # Visualize the results
            self.visualize_lane(
                cv_image,
                lane_center,
                all_white_masks if all_white_masks else white_lane_mask,
                all_yellow_masks if all_yellow_masks else yellow_lane_mask,
                all_red_masks if all_red_masks else red_stop_mask,
                center_white,
                center_yellow,
            )

        except Exception as e:
            rospy.logwarn(f"YOLO processing error: {e}. Using default values.")

    def visualize_lane(self, img, lane_center, white_masks=None, yellow_masks=None, red_masks=None, center_white=None, center_yellow=None):
        """
        Create a visualization with three panels:
        1. Original image
        2. Original image with segmentation overlays
        3. Original image with center points
        """
        h, w = img.shape[:2]

        full_width = w * 2
        vis_image = np.zeros((h, full_width, 3), dtype=np.uint8)

        # Panel 1: Original with segmentation overlays
        overlay = img.copy()

        # Erstelle eine kombinierte Maske für alle Segmentierungen
        combined_mask = np.zeros_like(img)

        # Verarbeite weiße Masken - können Liste oder einzelne Maske sein
        if isinstance(white_masks, list):
            for mask in white_masks:
                if mask is not None and mask.shape[:2] == img.shape[:2]:
                    combined_mask[mask > 0] = [255, 255, 255]  # Weiße Fahrspurmarkierung
        elif white_masks is not None and white_masks.shape[:2] == img.shape[:2]:
            combined_mask[white_masks > 0] = [255, 255, 255]  # Einzelne weiße Maske

        # Verarbeite gelbe Masken - können Liste oder einzelne Maske sein
        if isinstance(yellow_masks, list):
            for mask in yellow_masks:
                if mask is not None and mask.shape[:2] == img.shape[:2]:
                    # Gelb nur hinzufügen, wo noch keine andere Maske existiert
                    yellow_area = (mask > 0) & (combined_mask == 0).all(axis=2)
                    combined_mask[yellow_area] = [0, 255, 255]
        elif yellow_masks is not None and yellow_masks.shape[:2] == img.shape[:2]:
            yellow_area = (yellow_masks > 0) & (combined_mask == 0).all(axis=2)
            combined_mask[yellow_area] = [0, 255, 255]  # Gelbe Fahrspurmarkierung

        # Verarbeite rote Masken - können Liste oder einzelne Maske sein
        if isinstance(red_masks, list):
            for mask in red_masks:
                if mask is not None and mask.shape[:2] == img.shape[:2]:
                    # Rot nur hinzufügen, wo noch keine andere Maske existiert
                    red_area = (mask > 0) & (combined_mask == 0).all(axis=2)
                    combined_mask[red_area] = [0, 0, 255]
        elif red_masks is not None and red_masks.shape[:2] == img.shape[:2]:
            red_area = (red_masks > 0) & (combined_mask == 0).all(axis=2)
            combined_mask[red_area] = [0, 0, 255]  # Rote Stoppmarkierung

        # Original zu 40%, Maske zu 60%
        overlay = cv2.addWeighted(overlay, 0.4, combined_mask, 0.6, 0)

        red_stop_roi_window_height = 100
        red_stop_roi_window_width = 400
        red_stop_roi_window_y_start = img.shape[0] - 100 - red_stop_roi_window_height
        red_stop_roi_window_x_start = (img.shape[1] - red_stop_roi_window_width) // 2
        cv2.rectangle(
            overlay,
            (red_stop_roi_window_x_start, red_stop_roi_window_y_start),
            (red_stop_roi_window_x_start + red_stop_roi_window_width, red_stop_roi_window_y_start + red_stop_roi_window_height),
            (255, 255, 255),
            2,
        )

        vis_image[:, 0:w] = overlay

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

        vis_image[:, w : 2 * w] = center_vis

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis_image, "Segmentation", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(vis_image, "Lane Centers", (w + 10, 30), font, 1, (255, 255, 255), 2)

        # Display the visualization
        cv2.imshow("Lane Detection Visualization", vis_image)
        cv2.waitKey(1)


if __name__ == "__main__":

    node = DetectLaneNode(node_name="detect_lane_node")
    rospy.spin()
