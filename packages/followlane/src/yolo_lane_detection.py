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
        yolo_model_path = "/home/moritz_s/Documents/RKIM_1/Duckiebots/DuckieChallenge/packages/followlane/model/yolo_v11_seg_20250528.pt"  # Path to your lane segmentation model
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
        self._lane_viz_topic = f"/{self._vehicle_name}/detect/lane/visualization"
        self.pub_lane_viz = rospy.Publisher(self._lane_viz_topic, Image, queue_size=1)

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
        if self.counter % 3 != 0:
            self.counter += 1
            return
        else:
            self.counter += 1

        np_arr = np.frombuffer(image_msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img_orig = cv_image.copy()
        img_cropped = self.crop_img(img_orig)

        # YOLO Inferenz
        if not self.yolo_enabled:
            return

        try:
            results = self._model(cv_image)
        except Exception as e:
            rospy.logerr(f"YOLO inference error: {e}")
            return

        # Extrahiere YOLO-Masken
        yellow_lane_mask = None
        white_lane_mask = None

        if results is not None and hasattr(results[0], "masks") and results[0].masks is not None:
            for i, cls in enumerate(results[0].boxes.cls):
                mask = results[0].masks[i].data.cpu().numpy()
                mask = self.process_segmentation_mask(mask, cv_image.shape)
                if int(cls) == 1:  # Gelb
                    yellow_lane_mask = mask
                elif int(cls) == 0:  # Weiß
                    white_lane_mask = mask

        if yellow_lane_mask is None or white_lane_mask is None:
            rospy.logwarn("Not both yellow and white lane masks found.")
            return

        # Pixelkoordinaten
        ys_yellow, xs_yellow = np.where(yellow_lane_mask > 0)
        ys_white_all, xs_white_all = np.where(white_lane_mask > 0)

        if xs_yellow.size == 0 or xs_white_all.size == 0:
            rospy.logwarn("No yellow or white pixels found.")
            return

        # Weiß rechts der gelben Linie filtern
        x_g_mean = np.mean(xs_yellow)
        valid_indices = xs_white_all > x_g_mean
        xs_white = xs_white_all[valid_indices]
        ys_white = ys_white_all[valid_indices]

        if xs_white.size == 0:
            rospy.logwarn("No white pixels to the right of yellow line.")
            return

        # Fit: x = a*y + b
        fit_yellow = np.polyfit(ys_yellow, xs_yellow, 1)
        fit_white = np.polyfit(ys_white, xs_white, 1)

        y_eval = 50  # Höhe in der Bildmitte
        x_yellow = np.polyval(fit_yellow, y_eval)
        x_white = np.polyval(fit_white, y_eval)
        x_center = (x_yellow + x_white) / 2

        ###### VISUALISIERUNG ######
        vis_img = img_cropped.copy()

        cv2.circle(vis_img, (int(x_yellow), y_eval), 5, (0, 255, 255), -1)  # Gelb
        cv2.circle(vis_img, (int(x_white), y_eval), 5, (255, 255, 255), -1)  # Weiß
        cv2.circle(vis_img, (int(x_center), y_eval), 5, (0, 0, 255), -1)  # Spurmittelpunkt

        # Plot von YOLO (optional)
        seg_vis_img = results[0].plot() if hasattr(results[0], "plot") else cv_image

        # ROS Bild senden
        try:
            # Publish segmentation image
            viz_msg = self.bridge.cv2_to_imgmsg(seg_vis_img, "bgr8")
            self.pub_segmentation.publish(viz_msg)

            # Publish lane detection visualization
            lane_viz_msg = self.bridge.cv2_to_imgmsg(vis_img, "bgr8")
            self.pub_lane_viz.publish(lane_viz_msg)
        except Exception as e:
            rospy.logwarn(f"Error converting image: {e}")

        # Conditional debug display - only try to show if display is available
        try:
            cv2.imshow("Lane Detection", vis_img)
            cv2.imshow("YOLO Lane Segmentation", seg_vis_img)
            cv2.waitKey(1)
        except:
            pass  # Silently ignore if display isn't available

        # ROS Nachricht mit Zentren
        msg_centers = Float64MultiArray()
        msg_centers.data = [float(x_white), float(x_yellow), float(x_center)]
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
