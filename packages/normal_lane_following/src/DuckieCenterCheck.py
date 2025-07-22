#!/usr/bin/env python3

import os

import numpy as np
import rospkg
import rospy
import yaml
from cv_bridge import CvBridge
from default.msg import BoundingBox, BoundingBoxArray
from duckietown.dtros import DTROS, NodeType
from normal_lane_following.msg import MultiMaskGroups
from std_msgs.msg import Bool, Int32MultiArray


class DetectionCheckerNode(DTROS):
    def __init__(self, node_name):
        super(DetectionCheckerNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self.config = self._load_config()
        cc = self.config["center_check_duckie"]
        self.target_class_id = cc["target_class_id"]
        self.region = cc["region"]
        self.min_overlap_ratio = 0.4  # cc.get("min_overlap_ratio", 0.6)

        self.bridge = CvBridge()
        self._vehicle_name = os.environ["VEHICLE_NAME"]
        self._node_active = False

        # Subscribers & Publisher
        self.sub = rospy.Subscriber(f"/{self._vehicle_name}/detect/bounding_boxes", BoundingBoxArray, self.detection_callback, queue_size=1)
        self.sub_mode = rospy.Subscriber(f"/{self._vehicle_name}/current_mode", Int32MultiArray, self.cbControlMode, queue_size=1)
        self.sub_masks = rospy.Subscriber(f"/{self._vehicle_name}/detect/masks", MultiMaskGroups, self.mask_callback, queue_size=1)
        self.pub_in_region = rospy.Publisher(f"/{self._vehicle_name}/detect/in_region", Bool, queue_size=1)

        # Mask-Container
        self.yellow_masks = []
        self.white_masks = []
        self.dotted_masks = []

        rospy.loginfo(f"[DetectionCheckerNode] Monitoring duckie class {self.target_class_id}")

    def _load_config(self):
        pkg = rospkg.RosPack().get_path("default")
        cfg = os.path.join(pkg, "config", "processing_params.yaml")
        with open(cfg, "r") as f:
            data = yaml.safe_load(f)
            rospy.loginfo(f"Loaded config from {cfg}")
            return data

    def cbControlMode(self, msg: Int32MultiArray):
        self._node_active = msg.data[1] == 1
        if self._node_active:
            rospy.logwarn(f"{self._vehicle_name}: DetectionChecker active")

    def mask_callback(self, msg: MultiMaskGroups):
        if not self._node_active:
            return
        self.yellow_masks = [self.bridge.imgmsg_to_cv2(m, "mono8") for m in msg.yellow]
        self.white_masks = [self.bridge.imgmsg_to_cv2(m, "mono8") for m in msg.white]
        self.dotted_masks = [self.bridge.imgmsg_to_cv2(m, "mono8") for m in msg.dotted]

    def _line_positions_in_band(self, masks, y_low, y_high, find_min=True):
        """
        Hilfsfunktion: Aus einer Liste von Mono8-Masks nur
        Pixel in Zeilen [y_low, y_high] nehmen und
        das Min- oder Max-x zurückgeben.
        """
        xs_all = []
        for m in masks:
            ys, xs = np.where(m > 0)
            # filter auf den vertikalen Bereich
            mask_idx = (ys >= y_low) & (ys <= y_high)
            xs_band = xs[mask_idx]
            if xs_band.size > 0:
                xs_all.extend(xs_band.tolist())
        if not xs_all:
            return None
        return min(xs_all) if find_min else max(xs_all)

    def detection_callback(self, msg: BoundingBoxArray):
        if not self._node_active:
            return

        in_region = False

        for det in msg.boxes:
            if det.class_id != self.target_class_id:
                continue

            # vertikaler Bereich der Bounding Box ±10 %
            box_height = det.y_max - det.y_min
            margin = 0.1 * box_height
            y_low = max(0, int(det.y_min - margin))
            y_high = int(det.y_max + margin)

            # Linien-Positionen auf diesem Band bestimmen
            x_yellow = self._line_positions_in_band(self.yellow_masks, y_low, y_high, find_min=True)
            x_right_y = self._line_positions_in_band(self.white_masks, y_low, y_high, find_min=False)
            x_right_d = self._line_positions_in_band(self.dotted_masks, y_low, y_high, find_min=False)
            # wähle Max von weiß und dotted (falls beide vorhanden)
            x_right = None
            if x_right_y is not None and x_right_d is not None:
                x_right = max(x_right_y, x_right_d)
            elif x_right_y is not None:
                x_right = x_right_y
            elif x_right_d is not None:
                x_right = x_right_d

            # Fallback auf statische ROI, falls keine Linie
            if x_yellow is None or x_right is None:
                x_min_reg = self.region["x_min"]
                x_max_reg = self.region["x_max"]
            else:
                x_min_reg = x_yellow
                x_max_reg = x_right

            # Y-Mittelpunkt-Check (optional)
            y_center = 0.5 * (det.y_min + det.y_max)
            if not (self.region["y_min"] <= y_center <= self.region["y_max"]):
                continue

            # X-Überlappung
            box_w = det.x_max - det.x_min
            overlap = min(det.x_max, x_max_reg) - max(det.x_min, x_min_reg)
            overlap_ratio = max(0.0, overlap / box_w)

            if overlap_ratio >= self.min_overlap_ratio:
                in_region = True
                rospy.logwarn(f"Duckie in region: overlap {overlap_ratio:.2f} ≥ {self.min_overlap_ratio}")
                break

        self.pub_in_region.publish(Bool(data=in_region))


if __name__ == "__main__":
    node = DetectionCheckerNode(node_name="DetectionCheckerNode")
    rospy.spin()
