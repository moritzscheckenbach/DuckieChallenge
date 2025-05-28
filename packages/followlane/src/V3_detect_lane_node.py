#!/usr/bin/env python3

import os
import cv2
import numpy as np
import rospy
import yaml
from std_msgs.msg import Float64
from sensor_msgs.msg import CompressedImage
from duckietown.dtros import DTROS, NodeType

class DetectLaneNode(DTROS):
    def __init__(self, node_name):
        super(DetectLaneNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self.sub_image = rospy.Subscriber(self._camera_topic, CompressedImage, self.image_callback, queue_size=1)
        self.pub_center = rospy.Publisher(f"/{self._vehicle_name}/detect/lane", Float64, queue_size=1)

        self.counter = 0
        self.load_config("packages/followlane/config/detect_lane.yaml")

    def load_config(self, path):
        with open(path, "r") as f:
            self.conf = yaml.safe_load(f)

    def draw_roi_lines(self, image):
        x_alt, y_alt = None, None
        for key in ["top_left", "top_right", "bottom_left", "bottom_right", "top_left"]:
            x = self.conf["lane_image"][f"{key}_x"]
            y = self.conf["lane_image"][f"{key}_y"]
            if x_alt is not None and y_alt is not None:
                cv2.line(image, (x_alt, y_alt), (x, y), (255, 255, 255), 2)
            x_alt, y_alt = x, y

    def apply_roi_mask(self, image):
        mask = np.zeros_like(image)
        pts = np.array([[
            (self.conf["lane_image"]["top_left_x"], self.conf["lane_image"]["top_left_y"]),
            (self.conf["lane_image"]["top_right_x"], self.conf["lane_image"]["top_right_y"]),
            (self.conf["lane_image"]["bottom_left_x"], self.conf["lane_image"]["bottom_left_y"]),
            (self.conf["lane_image"]["bottom_right_x"], self.conf["lane_image"]["bottom_right_y"]),
        ]], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255)
        return cv2.bitwise_and(image, mask)

    def average_slope_intercept(self, lines):
        left_lines = []
        right_lines = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 - x1 == 0:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.3:
                    continue
                intercept = y1 - slope * x1
                if slope < 0:
                    left_lines.append((slope, intercept))
                else:
                    right_lines.append((slope, intercept))

        def average_line(line_group):
            if not line_group:
                return []
            slope_avg = np.mean([s for s, _ in line_group])
            intercept_avg = np.mean([i for _, i in line_group])
            y1 = self.conf["lane_image"]["bottom_left_y"]
            y2 = self.conf["lane_image"]["top_left_y"]
            x1 = int((y1 - intercept_avg) / slope_avg)
            x2 = int((y2 - intercept_avg) / slope_avg)
            return [[[x1, y1, x2, y2]]]

        return average_line(left_lines), average_line(right_lines)


    def compute_lane_center(self, left_lines, right_lines, width):
        def avg_x(lines):
            xs = [x for line in lines for x1, _, x2, _ in line for x in [x1, x2]]
            return np.mean(xs) if xs else None

        x_left = avg_x(left_lines)
        x_right = avg_x(right_lines)

        if x_left and x_right:
            return (x_left + x_right) / 2
        elif x_left:
            return x_left + width * 0.25
        elif x_right:
            return x_right - width * 0.25
        else:
            return width / 2

    def image_callback(self, msg):
        if self.counter % 2 != 0:
            self.counter += 1
            return
        self.counter += 1

        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        height, width = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        masked_edges = self.apply_roi_mask(edges)
        debug_img = frame.copy()
        self.draw_roi_lines(debug_img)

        lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=150)
        left_lines, right_lines = [], []

        if lines is not None:
            left_lines, right_lines = self.average_slope_intercept(lines)
            for line in left_lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # blau = links
            for line in right_lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # rot = rechts

        lane_center = self.compute_lane_center(left_lines, right_lines, width)
        image_center = width / 2
        error = lane_center - image_center

        # Visualisierung der Spurmitte und Bildmitte
        cv2.line(debug_img, (int(lane_center), 0), (int(lane_center), height), (0, 255, 0), 2)
        cv2.line(debug_img, (int(image_center), 0), (int(image_center), height), (0, 255, 255), 1)

        msg_out = Float64()
        msg_out.data = error
        self.pub_center.publish(msg_out)

        debug_resized = cv2.resize(debug_img, None, fx=1.5, fy=1.5)
        cv2.imshow("Lane Detection (Hough+ROI)", debug_resized)
        cv2.waitKey(1)

if __name__ == '__main__':
    node = DetectLaneNode(node_name='detect_lane_node')
    rospy.spin()
    cv2.destroyAllWindows()
