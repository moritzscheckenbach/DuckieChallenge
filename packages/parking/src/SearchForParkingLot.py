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
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Float64, Int32MultiArray, String


class SearchForParkingLot(DTROS):
    def __init__(self, node_name):
        super(SearchForParkingLot, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.X_QUERY = 220
        self._vehicle_name = os.environ.get("VEHICLE_NAME", "")

        self.config = self._load_config()

        self.target_class_id = self.config["parking"]["target_class_id"]  # Parkplatz-Klasse
        self.duckie_class_id = self.config["parking"]["duckie_class_id"]  # Enten-Klasse
        self.bot_class_id = self.config["parking"]["bot_class_id"]  # Roboter-Klasse
        self.region = self.config["parking"]["region"]  # [x_min, x_max, y_min, y_max]

        self._node_active = False
        self.parking_occupied = False

        # rospy.logwarn(f"{self._vehicle_name}: ParkingCheckNode initialized")
        self.sub_masks = rospy.Subscriber(f"/{self._vehicle_name}/detect/masks", MultiMaskGroups, self.Mask_Callback, queue_size=1)
        self.bridge = CvBridge()

        self.sub = rospy.Subscriber(f"/{self._vehicle_name}/detect/bounding_boxes", BoundingBoxArray, self.detection_callback, queue_size=1)
        self.sub_control_mode = rospy.Subscriber(f"/{self._vehicle_name}/current_mode", Int32MultiArray, self.cbControlMode, queue_size=1)

        self.pub_parking_possible = rospy.Publisher(f"/{self._vehicle_name}/parking/possible", Bool, queue_size=1)
        self.pub_parking_occupied = rospy.Publisher(f"/{self._vehicle_name}/parking/occupied", Bool, queue_size=1)

        rospy.loginfo(f"[SearchForParkingLot] Monitoring class {self.target_class_id} in region {self.region}")

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
        if msg.data[8] == 1:
            rospy.sleep(5)
            self._node_active = True
            rospy.logwarn(f"{self._vehicle_name}: Parking check active")
        else:
            self._node_active = False
            self.parking_occupied = False

    def is_in_region(self, box):
        x_center = (box.x_min + box.x_max) / 2.0
        y_center = (box.y_min + box.y_max) / 2.0

        return self.region["x_min"] <= x_center <= self.region["x_max"] and self.region["y_min"] <= y_center <= self.region["y_max"]

    def iou(self, boxA, boxB):
        # Intersection over Union zur Überlappungsprüfung
        xA = max(boxA.x_min, boxB.x_min)
        yA = max(boxA.y_min, boxB.y_min)
        xB = min(boxA.x_max, boxB.x_max)
        yB = min(boxA.y_max, boxB.y_max)

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA.x_max - boxA.x_min) * (boxA.y_max - boxA.y_min)
        boxBArea = (boxB.x_max - boxB.x_min) * (boxB.y_max - boxB.y_min)
        iou = interArea / float(boxAArea + boxBArea - interArea) if float(boxAArea + boxBArea - interArea) > 0 else 0

        rospy.logwarn(f"IOU between {boxA.class_id} and {boxB.class_id}: {iou:.2f}")
        return iou

    def Mask_Callback(self, msg: MultiMaskGroups):
        if self._node_active:
            # Wandelt sensor_msgs/Image[] in OpenCV-Bilder um
            self.yellow_masks = [self.bridge.imgmsg_to_cv2(m, desired_encoding="mono8") for m in msg.yellow]
            dotted_masks = [self.bridge.imgmsg_to_cv2(m, desired_encoding="mono8") for m in msg.dotted]

            # **Nur noch feststellen, ob überhaupt dotted-Masken übertragen wurden**
            self.dotted_detected = len(dotted_masks) > 0

        # hier deine Wunsch-Spalte eintragen

    def has_yellow_at_x(self, mask, x):
        """True, falls irgendwo in Spalte x gelbe Pixel (>0) stehen."""
        return np.any(mask[:, x] > 0)

        # gibt es in *irgendeiner* Yellow-Maske gelb an X_QUERY?

    def detection_callback(self, msg):
        if not self._node_active:
            return

        boxes = msg.boxes
        AREA_THRESHOLD = 25000
        large_parking_possible = False

        # 1) Suche Parkplatz-BoundingBoxes
        parking_boxes = [b for b in boxes if b.class_id == self.target_class_id]
        rospy.logwarn(f"Found {len(parking_boxes)} parking boxes")
        # 2) Finde den ersten großen Parkplatz
        large_parking = next((b for b in parking_boxes if (b.x_max - b.x_min) * (b.y_max - b.y_min) > AREA_THRESHOLD), None)
        rospy.logwarn(f"Found large parking: {large_parking}")

        if large_parking:
            rospy.logwarn(f"PARKPLATZ IST GROOOOOOOOOOSS")
            # 3) Überprüfe Region
            if self.is_in_region(large_parking):
                rospy.logwarn(f"PARKPLATZ IST IN REGION")
                # 4) dotted-Masken vorhanden?
                if hasattr(self, "dotted_detected") and self.dotted_detected:
                    rospy.logwarn(f"SEHE DOTTE LINE")
                    # 5) x-Koordinate ermitteln
                    x_center = int((large_parking.x_min + large_parking.x_max) / 2.0)

                    # 6) gibt es gelb an dieser x-Koordinate?
                    has_yellow = any(self.has_yellow_at_x(mask, x_center) for mask in self.yellow_masks)

                    if not has_yellow:
                        rospy.logwarn(f"KEINE GELBE LINIE DAZWISCHEN")
                        large_parking_possible = True

                        # 7) Hindernisse auf großem Parkplatz prüfen
                        obstacle_boxes = [b for b in boxes if b.class_id in [self.duckie_class_id, self.bot_class_id]]
                        for obstacle in obstacle_boxes:
                            rospy.logwarn(f"SChaue ob belegt BELEEEEEEEEEEEEGT")
                            if self.iou(large_parking, obstacle) > 0:
                                rospy.logwarn(f"PARKPLATZ IST BELEEEEEEEEEEEEGT")
                                self.parking_occupied = True
                                break

        # 8) Ergebnis publizieren
        rospy.logwarn(f"Large parking possible: {large_parking_possible}, Parking occupied: {self.parking_occupied}")

        if self._node_active == True:
            rospy.logwarn(f"publishing parking possible and occupied or not")
            self.pub_parking_possible.publish(Bool(large_parking_possible))
            self.pub_parking_occupied.publish(Bool(self.parking_occupied))


if __name__ == "__main__":
    node = SearchForParkingLot(node_name="SearchForParkingLot")
    rospy.spin()
