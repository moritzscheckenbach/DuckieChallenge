#!/usr/bin/env python3

import os
from enum import Enum

import cv2
import numpy as np
import rospy
import yaml
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64


class DetectLaneNode(DTROS):
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(DetectLaneNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        self.load_conf("packages/followlane/config/detect_lane.yaml")
        self._vehicle_name = os.environ["VEHICLE_NAME"]
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"

        self.sub_image_original = rospy.Subscriber(self._camera_topic, CompressedImage, self.cbFindLane, queue_size=1)

        self.pub_lane = rospy.Publisher(f"/{self._vehicle_name}/detect/lane", Float64, queue_size=1)

        self.counter = 0

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

    def cbFindLane(self, image_msg):
        if self.counter % 3 != 0:
            self.counter += 1
            return
        else:
            self.counter += 1

        # Decode Image
        np_arr = np.frombuffer(image_msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img = self.crop_img(cv_image)

        # Bild vorverarbeiten
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Dynamische Lichtanpassung (Histogramm-Equalisierung auf V-Kanal)
        h, s, v = cv2.split(hsv)
        v_eq = cv2.equalizeHist(v)
        hsv_eq = cv2.merge((h, s, v_eq))

        # Masken anwenden
        mask_yellow = cv2.inRange(hsv_eq, (self.hue_yellow_l, self.saturation_yellow_l, self.lightness_yellow_l), (self.hue_yellow_h, self.saturation_yellow_h, self.lightness_yellow_h))

        mask_white = cv2.inRange(hsv_eq, (self.hue_white_l, self.saturation_white_l, self.lightness_white_l), (self.hue_white_h, self.saturation_white_h, self.lightness_white_h))

        # Berechne Schwerpunkte (Momente)
        def compute_center(mask):
            moments = cv2.moments(mask)
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])
                return cx
            return None

        center_yellow = compute_center(mask_yellow)
        center_white = compute_center(mask_white)

        # Debug: Anzeige und Logging (falls gewünscht)
        rospy.loginfo_throttle(1.0, f"White center: {center_white}, Yellow center: {center_yellow}")

        # Fallbacks bei schlechter Erkennung
        if center_white is None and center_yellow is None:
            desired_center = 500  # Default fallback
        elif center_white is None:
            desired_center = center_yellow + 100
        elif center_yellow is None:
            desired_center = center_white - 100
        else:
            desired_center = (center_white + center_yellow) / 2

        # Publish Ergebnis
        msg = Float64()
        msg.data = desired_center
        self.pub_lane.publish(msg)

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
