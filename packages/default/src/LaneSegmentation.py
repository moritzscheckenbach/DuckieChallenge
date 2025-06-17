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
from std_msgs.msg import Bool, Float64
from ultralytics import YOLO


class LaneSegmentation(DTROS):
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(LaneSegmentation, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        self._vehicle_name = os.environ["VEHICLE_NAME"]

        self.image_height = 480  # Default image height before cropping

        # Load configuration parameters
        self.config = self._load_config()

        self.Xth_frame = self.config["processing"]["use_every_Xth_frame"]  # Process every Xth frame
        self.crop_height_percentage = self.config["processing"]["crop_height_percentage"]  # Percentage of the image height to crop from the top

        self.pub_all_masks = rospy.Publisher(f"/{self._vehicle_name}/detect/masks", MultiMaskGroups, queue_size=1)
        self.bridge = CvBridge()
        self.pub_red_stop_detected = rospy.Publisher(f"/{self._vehicle_name}/redstop_detected", Bool, queue_size=1)

        self.currently_red = False

        # Initialize YOLO model for lane segmentation
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("default")
        yolo_model_path = os.path.join(package_path, "src", "model", "yolo_v11_lane_seg_20250528.pt")
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

        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self.sub_image_original = rospy.Subscriber(self._camera_topic, CompressedImage, self.SegmentImage, queue_size=1)

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

    def crop_img(self, img):
        img = img.copy()
        h, w = img.shape[:2]
        crop_height = int(h * self.crop_height_percentage)  # Crop X% from the top
        img = img[crop_height:, :]
        self.image_height = img.shape[0]
        rospy.loginfo(f"image size: height:{img.shape[0]}, width:{img.shape[1]}")

        return img

        # NOTE: If needed place bird's eye view transformation here

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

    def publish_masks(self, white_masks, yellow_masks, red_masks, dotted_masks):
        self.check_red_stop(red_masks)
        msg = MultiMaskGroups()
        now = rospy.Time.now()

        def convert(mask_list):
            imgs = [self.bridge.cv2_to_imgmsg(m, encoding="mono8") for m in mask_list]
            for im in imgs:
                im.header.stamp = now
            return imgs

        msg.white = convert(white_masks)
        msg.yellow = convert(yellow_masks)
        msg.red = convert(red_masks)
        msg.dotted = convert(dotted_masks)

        self.pub_all_masks.publish(msg)

    def check_red_stop(self, red_masks):
        """
        Check if the red stop mask indicates a stop condition.

        Logic:
        - If there was a red mask detected in the previous frame but not in the current frame,
        publish True on the redstop_detected topic.
        - Otherwise do nothing.
        """
        # Initialize a combined mask
        red_mask = None

        # Walk through all the red masks in the list and combine them
        if red_masks:
            for mask in red_masks:
                if mask is not None:
                    if red_mask is None:
                        # Initialize the combined mask with the first valid mask
                        red_mask = mask.copy()
                    else:
                        # Combine the masks using logical OR
                        red_mask = np.logical_or(red_mask, mask).astype(np.uint8)

        if red_mask is None:
            # No mask provided for checking
            has_red_now = False
        else:
            # Get the ROI window dimensions and position from config
            roi_height = self.config["red_stop"]["window_height"]
            roi_width = self.config["red_stop"]["window_width"]
            bottom_offset = self.config["red_stop"]["window_bottom_offset"]

            # Calculate ROI position
            y_start = red_mask.shape[0] - bottom_offset - roi_height
            x_start = (red_mask.shape[1] - roi_width) // 2

            # Make sure the ROI is within image bounds
            y_start = max(0, y_start)
            y_end = min(red_mask.shape[0], y_start + roi_height)
            x_start = max(0, x_start)
            x_end = min(red_mask.shape[1], x_start + roi_width)

            # Extract window from the mask
            window_mask = red_mask[y_start:y_end, x_start:x_end]

            # Check if mask has red pixels in ROI
            if window_mask.size > 0:
                # Calculate percentage of red pixels in the ROI
                mask_percentage = np.sum(window_mask > 0) / window_mask.size * 100
                has_red_now = True
                rospy.logdebug(f"Red mask coverage in ROI: {mask_percentage:.2f}%")
            else:
                has_red_now = False

        # Handle transition: if we HAD red but NOW don't, publish True
        if self.currently_red and not has_red_now:
            rospy.loginfo("Red mask disappeared from ROI - publishing stop signal")
            red_stop_msg = Bool()
            red_stop_msg.data = True
            self.pub_red_stop_detected.publish(red_stop_msg)

        # Update state for next frame
        self.currently_red = has_red_now

    def SegmentImage(self, image_msg):
        if self.counter % self.Xth_frame != 0:
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

            white_lane_mask = None
            yellow_lane_mask = None
            red_stop_mask = None
            dotted_mask = None

            all_white_masks = []
            all_yellow_masks = []
            all_red_masks = []
            all_dotted_masks = []

            # Behavior selection based on detected classes
            if results is not None and hasattr(results[0], "masks") and results[0].masks is not None:
                # Get all detected classes
                detected_classes = results[0].boxes.cls.cpu().numpy().astype(int)
                rospy.logwarn(f"Detected classes: {detected_classes}")

                if 1 in detected_classes:
                    # Extract white lane class
                    white_indices = [i for i, cls in enumerate(results[0].boxes.cls) if int(cls) == 1]
                    raw_white_lane_mask = results[0].masks[white_indices[0]].data.cpu().numpy()
                    white_lane_mask = self.process_segmentation_mask(raw_white_lane_mask, cv_image.shape)

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

                if 3 in detected_classes:
                    # Extract dotted stop class
                    dotted_indices = [i for i, cls in enumerate(results[0].boxes.cls) if int(cls) == 3]
                    if dotted_indices:
                        raw_dotted_mask = results[0].masks[dotted_indices[0]].data.cpu().numpy()
                        dotted_mask = self.process_segmentation_mask(raw_dotted_mask, cv_image.shape)

                    # Für die Visualisierung: Sammeln aller roten Masken
                    for idx in dotted_indices:
                        raw_mask = results[0].masks[idx].data.cpu().numpy()
                        mask = self.process_segmentation_mask(raw_mask, cv_image.shape)
                        if mask is not None:
                            all_dotted_masks.append(mask)

            self.publish_masks(all_white_masks, all_yellow_masks, all_red_masks, all_dotted_masks)

            # Visualize the results
            self.visualize_lane(
                cv_image,
                all_white_masks if all_white_masks else white_lane_mask,
                all_yellow_masks if all_yellow_masks else yellow_lane_mask,
                all_red_masks if all_red_masks else red_stop_mask,
                all_dotted_masks if all_dotted_masks else dotted_mask,
            )

        except Exception as e:
            rospy.logwarn(f"YOLO processing error: {e}. Using default values.")

    def visualize_lane(self, img, white_masks=None, yellow_masks=None, red_masks=None, dotted_masks=None):

        h, w = img.shape[:2]

        full_width = w
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

        # Verarbeite Dotted Masken - können Liste oder einzelne Maske sein
        if isinstance(dotted_masks, list):
            for mask in dotted_masks:
                if mask is not None and mask.shape[:2] == img.shape[:2]:
                    # Rot nur hinzufügen, wo noch keine andere Maske existiert
                    dotted_area = (mask > 0) & (combined_mask == 0).all(axis=2)
                    combined_mask[dotted_area] = [0, 0, 255]
        elif dotted_masks is not None and dotted_masks.shape[:2] == img.shape[:2]:
            dotted_area = (dotted_masks > 0) & (combined_mask == 0).all(axis=2)
            combined_mask[dotted_area] = [0, 0, 255]  # Dotted Markierung

        # Original zu 40%, Maske zu 60%
        overlay = cv2.addWeighted(overlay, 0.4, combined_mask, 0.6, 0)

        vis_image[:, 0:w] = overlay

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis_image, "Segmentation", (10, 30), font, 1, (255, 255, 255), 2)

        # Display the visualization
        cv2.imshow("Lane Detection Visualization", vis_image)
        cv2.waitKey(1)


if __name__ == "__main__":

    node = LaneSegmentation(node_name="LaneSegmentation")
    rospy.spin()
