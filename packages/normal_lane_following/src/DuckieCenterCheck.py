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
        self.park_class_id = 2
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

        # Sammle alle Enten-Detektionen und finde die größte (nächeste)
        duckie_boxes = [det for det in msg.boxes if det.class_id == self.target_class_id]

        if not duckie_boxes:
            self.pub_in_region.publish(Bool(data=False))
            return

        # Finde die größte Ente (größte Bounding Box Fläche = nächeste)
        largest_duckie = max(duckie_boxes, key=lambda det: (det.x_max - det.x_min) * (det.y_max - det.y_min))

        # Sammle alle Parkplatz-Bounding-Boxen
        park_boxes = [det for det in msg.boxes if det.class_id == self.park_class_id]

        # Prüfe nur die größte/nächeste Ente
        det = largest_duckie

        # Prüfe ob die Ente zu 80% oder mehr mit einem Parkplatz überlappt
        duckie_ignored = False
        duckie_area = (det.x_max - det.x_min) * (det.y_max - det.y_min)

        for park_box in park_boxes:
            # Berechne Überlappung zwischen Enten- und Parkplatz-Box
            overlap_x_min = max(det.x_min, park_box.x_min)
            overlap_x_max = min(det.x_max, park_box.x_max)
            overlap_y_min = max(det.y_min, park_box.y_min)
            overlap_y_max = min(det.y_max, park_box.y_max)

            if overlap_x_min < overlap_x_max and overlap_y_min < overlap_y_max:
                overlap_area = (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min)
                overlap_ratio = overlap_area / duckie_area

                if overlap_ratio >= 0.8:
                    rospy.logwarn(f"Größte Duckie ignoriert: {overlap_ratio:.2f} Überlappung mit Parkplatz")
                    duckie_ignored = True
                    break

        if not duckie_ignored:

            # vertikaler Bereich der Bounding Box ±10 %
            box_height = det.y_max - det.y_min
            margin = 0.1 * box_height
            y_low = max(0, int(det.y_min - margin))
            y_high = int(det.y_max + margin)

            # Prüfe ob Ente auf meiner Fahrbahn ist (gelbe Linie LINKS von der Ente)
            x_yellow_left = None
            x_white_right = None

            # Suche gelbe Linie LINKS von der Ente
            for m in self.yellow_masks:
                ys, xs = np.where(m > 0)
                mask_idx = (ys >= y_low) & (ys <= y_high) & (xs < det.x_min)  # links von BBox
                xs_band = xs[mask_idx]
                if xs_band.size > 0:
                    x_yellow_left = max(xs_band)  # nächste zur BBox
                    break

            # Suche weiße/gestrichelte Linie RECHTS von der Ente
            # Erst gestrichelte Masken prüfen (haben Priorität)
            for m in self.dotted_masks:
                ys, xs = np.where(m > 0)
                mask_idx = (ys >= y_low) & (ys <= y_high) & (xs > det.x_max)  # rechts von BBox
                xs_band = xs[mask_idx]
                if xs_band.size > 0:
                    x_white_right = min(xs_band)  # nächste zur BBox
                    break

            # Falls keine gestrichelte Linie, dann weiße prüfen
            if x_white_right is None:
                for m in self.white_masks:
                    ys, xs = np.where(m > 0)
                    mask_idx = (ys >= y_low) & (ys <= y_high) & (xs > det.x_max)  # rechts von BBox
                    xs_band = xs[mask_idx]
                    if xs_band.size > 0:
                        x_white_right = min(xs_band)  # nächste zur BBox
                        break

            # Ente ist auf MEINER Fahrbahn wenn: gelbe Linie LINKS und weiße/gestrichelte RECHTS
            if x_yellow_left is not None and x_white_right is not None:
                in_region = True
                rospy.logwarn(f"Ente auf meiner Fahrbahn erkannt: gelb links bei x={x_yellow_left}, weiß/gestrichelt rechts bei x={x_white_right}")
            else:
                # Debug: Warum nicht erkannt?
                if x_yellow_left is None and x_white_right is not None:
                    rospy.loginfo("Ente auf Gegenfahrbahn: keine gelbe Linie links gefunden")
                elif x_yellow_left is not None and x_white_right is None:
                    rospy.loginfo("Ente zwischen Spuren: keine weiße/gestrichelte Linie rechts gefunden")
                else:
                    rospy.loginfo("Ente Position unbekannt: keine passenden Linien gefunden")

        self.pub_in_region.publish(Bool(data=in_region))


if __name__ == "__main__":
    node = DetectionCheckerNode(node_name="DetectionCheckerNode")
    rospy.spin()
