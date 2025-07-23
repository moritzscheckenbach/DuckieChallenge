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
from std_msgs.msg import Bool, Float64, Int32, Int32MultiArray, String


class DetectionCheckerNode(DTROS):
    def __init__(self, node_name):
        super(DetectionCheckerNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        # rospy.logwarn("Hallo, ich bin der DetectionCheckerNode!")
        self.config = self._load_config()

        self.target_class_id = self.config["center_check_duckie"]["target_class_id"]
        self.park_class_id = 2  # Parkplatz Klasse ID
        self.region = self.config["center_check_duckie"]["region"]

        self.bridge = CvBridge()
        self._vehicle_name = os.environ["VEHICLE_NAME"]
        self._node_active = False

        # Mask-Container für Linienanalyse
        self.yellow_masks = []

        self.sub = rospy.Subscriber(f"/{self._vehicle_name}/detect/bounding_boxes", BoundingBoxArray, self.detection_callback, queue_size=1)
        self.sub_control_mode = rospy.Subscriber(f"/{self._vehicle_name}/current_mode", Int32MultiArray, self.cbControlMode, queue_size=1)
        self.sub_masks = rospy.Subscriber(f"/{self._vehicle_name}/detect/masks", MultiMaskGroups, self.mask_callback, queue_size=1)

        self.pub_in_region = rospy.Publisher(f"/{self._vehicle_name}/detect/in_region", Bool, queue_size=1)

        rospy.loginfo(f"[DetectionCheckerNode] Läuft. Überwacht Klasse {self.target_class_id} im Bereich {self.region}")

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

    def cbControlMode(self, msg: Int32MultiArray):
        if msg.data[1] == 1:
            self._node_active = True
            rospy.logwarn(f"{self._vehicle_name}: DuckieCenterCheck is now active")
        else:
            self._node_active = False

    def mask_callback(self, msg: MultiMaskGroups):
        """Callback für Masken-Daten (Linien)"""
        if not self._node_active:
            return
        self.yellow_masks = [self.bridge.imgmsg_to_cv2(m, "mono8") for m in msg.yellow]

    def _is_duckie_in_parking(self, duckie_box, park_boxes):
        """Prüft ob eine Ente zu 80% oder mehr mit einem Parkplatz überlappt"""
        # duckie_area = (duckie_box.x_max - duckie_box.x_min) * (duckie_box.y_max - duckie_box.y_min)

        for park_box in park_boxes:
            # Berechne Überlappung
            overlap_x_min = max(duckie_box.x_min, park_box.x_min)
            overlap_x_max = min(duckie_box.x_max, park_box.x_max)
            overlap_y_min = max(duckie_box.y_min, park_box.y_min)
            overlap_y_max = min(duckie_box.y_max, park_box.y_max)

            if overlap_x_min < overlap_x_max and overlap_y_min < overlap_y_max:
                overlap_area = (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min)
                overlap_ratio = overlap_area / self.duckie_area

                if overlap_ratio >= 0.8:
                    return True
        return False

    def _is_duckie_on_wrong_side(self, duckie_box):
        """Prüft ob eine Ente links von der gelben Linie ist (Gegenfahrbahn) oder über der gelben Linie liegt"""
        if not self.yellow_masks:
            return False  # Keine Masken verfügbar

        # Vertikaler Bereich der Ente ±10%
        box_height = duckie_box.y_max - duckie_box.y_min
        margin = 0.1 * box_height
        y_low = max(0, int(duckie_box.y_min - margin))
        y_high = int(duckie_box.y_max + margin)

        for m in self.yellow_masks:
            ys, xs = np.where(m > 0)
            # Nur Maskenpixel auf (ungefähr) gleicher Höhe wie die Ente betrachten
            same_height_idx = (ys >= duckie_box.y_min) & (ys <= duckie_box.y_max)

            # 1. Prüfe ob gelbe Linie rechts von der Ente ist (= Ente auf Gegenfahrbahn)
            mask_idx_right = same_height_idx & (ys >= y_low) & (ys <= y_high) & (xs > duckie_box.x_max)
            if xs[mask_idx_right].size > 0:
                rospy.loginfo("Ente auf Gegenfahrbahn erkannt (gelbe Linie rechts)")
                return True

            # 2. Prüfe ob gelbe Linie unter der Ente liegt (= Ente überquert gelbe Linie)
            mask_idx_under = same_height_idx & (ys >= y_low) & (ys <= y_high) & (xs >= duckie_box.x_min) & (xs <= duckie_box.x_max)
            if xs[mask_idx_under].size > 0:
                rospy.loginfo("Ente über gelber Linie erkannt")
                return True

        return False

    def detection_callback(self, msg):
        if not self._node_active:
            return

        in_region = False
        area_threshold = 1000

        # Sammle alle Parkplatz-Bounding-Boxen
        park_boxes = [det for det in msg.boxes if det.class_id == self.park_class_id]

        for detection in msg.boxes:
            if detection.class_id == self.target_class_id:

                # Prüfe Confidence-Threshold (50%)
                if detection.confidence < 0.5:
                    rospy.loginfo(f"Ente mit zu niedriger Confidence ignoriert: {detection.confidence:.2f} < 0.5")
                    continue

                self.duckie_area = (detection.x_max - detection.x_min) * (detection.y_max - detection.y_min)
                rospy.logwarn(f"Ente BBox area: {self.duckie_area}, Confidence: {detection.confidence:.2f}")

                # 0. Prüfe ob Ente Groß genug ist
                if self.duckie_area < area_threshold:
                    # rospy.logwarn(f"Ente BBox area {duckie_area} < threshold {area_threshold} -- Detection übersprungen")
                    continue

                # 1. Prüfe ob Ente im Parkplatz ist
                if self._is_duckie_in_parking(detection, park_boxes):
                    rospy.loginfo("Ente im Parkplatz - wird ignoriert")
                    continue

                # 2. Prüfe ob Ente auf Gegenfahrbahn ist (links von gelber Linie)
                if self._is_duckie_on_wrong_side(detection):
                    rospy.loginfo("Ente auf Gegenfahrbahn - wird ignoriert")
                    continue

                # 3. Normale Region-Prüfung
                x_min, y_min = detection.x_min, detection.y_min
                x_max, y_max = detection.x_max, detection.y_max

                if (self.region["x_min"] <= x_min <= self.region["x_max"] and self.region["y_min"] <= y_min <= self.region["y_max"]) or (
                    self.region["x_min"] <= x_max <= self.region["x_max"] and self.region["y_min"] <= y_max <= self.region["y_max"]
                ):
                    in_region = True
                    rospy.logwarn(f"Ente der Klasse {self.target_class_id} erkannt im Bereich: {self.region}")
                    break

        self.pub_in_region.publish(Bool(data=in_region))


if __name__ == "__main__":
    node = DetectionCheckerNode(node_name="DetectionCheckerNode")
    rospy.spin()
