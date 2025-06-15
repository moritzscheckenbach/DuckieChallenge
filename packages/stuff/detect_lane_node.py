#!/usr/bin/env python3

import os
from enum import Enum

import cv2
import numpy as np
import rospy
import yaml
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64MultiArray


class DetectLaneNode(DTROS):
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(DetectLaneNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        self.load_conf("packages/followlane/config/detect_lane.yaml")
        self._vehicle_name = os.environ["VEHICLE_NAME"]
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"

        self.sub_image_original = rospy.Subscriber(self._camera_topic, CompressedImage, self.cbFindLane, queue_size=1)

        self.pub_lane = rospy.Publisher(f"/{self._vehicle_name}/detect/lane", Float64MultiArray, queue_size=1)

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

        # Write your own Code for Lane detection here
        # This is only a basic example to get some inspiration from

        np_arr = np.frombuffer(image_msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img = self.crop_img(cv_image)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask_yellow = cv2.inRange(
            hsv,
            (self.hue_yellow_l, self.saturation_yellow_l, self.lightness_yellow_l),
            (self.hue_yellow_h, self.saturation_yellow_h, self.lightness_yellow_h),
        )

        mask_white = cv2.inRange(
            hsv,
            (self.hue_white_l, self.saturation_white_l, self.lightness_white_l),
            (self.hue_white_h, self.saturation_white_h, self.lightness_white_h),
        )

        ys_white, xs_white = np.where(mask_white != 0)
        ys_yellow, xs_yellow = np.where(mask_yellow != 0)

        center_white = np.mean(xs_white) if xs_white.size > 0 else 100
        center_yellow = np.mean(xs_yellow) if xs_yellow.size > 0 else 900

        if np.isnan(center_white):
            center_white = 100

        if np.isnan(center_yellow):
            center_yellow = 900

        calculated_center = (center_white + center_yellow) / 2

        ###### image visualization ######
        if not np.isnan(center_white):
            cv2.circle(img, (int(center_white), 50), 5, (255, 0, 0), -1)
        if not np.isnan(center_yellow):
            cv2.circle(img, (int(center_yellow), 50), 5, (0, 255, 0), -1)
        cv2.circle(img, (int(calculated_center), 50), 5, (0, 0, 255), -1)
        cv2.imshow("Lane Detection", img)
        cv2.waitKey(1)

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
