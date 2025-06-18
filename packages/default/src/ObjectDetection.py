#!/usr/bin/env python3

import os

import cv2
import numpy as np
import rospkg
import rospy
import torch
import yaml
from cv_bridge import CvBridge
from default.msg import BoundingBox, BoundingBoxArray
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from ultralytics import YOLO


class ShowCameraNode(DTROS):
    """
    DTROS-Node, das das Kamerabild abonniert, YOLO-Detections ausführt
    und die Ergebnisse als Bild sowie als BoundingBoxArray publiziert.
    """

    def __init__(self, node_name):
        super(ShowCameraNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        self._vehicle_name = os.environ.get("VEHICLE_NAME", "")
        if not self._vehicle_name:
            rospy.logwarn("VEHICLE_NAME nicht gesetzt, verwende Standard-Topic")

        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self.bridge = CvBridge()

        # Device bestimmen (GPU oder CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"Verwendetes Gerät: {self.device}")

        # YOLO-Modell laden
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("default")
        yolo_model_path = os.path.join(package_path, "src", "model", "yolo_v11_obj_dect_20250610.pt")

        if os.path.exists(yolo_model_path):
            self.model = YOLO(yolo_model_path)
            self.model.to(self.device)
            rospy.loginfo(f"YOLO-Modell geladen: {yolo_model_path} auf Gerät: {self.device}")
        else:
            rospy.logerr(f"YOLO-Modell nicht gefunden: {yolo_model_path}")
            self.model = None

        # Publisher für BoundingBoxes
        self.pub_boxes = rospy.Publisher(f"/{self._vehicle_name}/detect/bounding_boxes", BoundingBoxArray, queue_size=1)

        # Bild-Subscriber
        self.sub_image = rospy.Subscriber(self._camera_topic, CompressedImage, self.cb_display_image, queue_size=1)
        rospy.loginfo(f"[{self.node_name}] Abonniert: {self._camera_topic}")

        self.counter = 0

        self.config = self._load_config()
        if self.config:
            self.Xth_frame = self.config["processing"]["use_every_Xth_frame_2"]
        else:
            self.Xth_frame = 1  # Fallback

    def _load_config(self):
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("default")
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

    def cb_display_image(self, image_msg):
        if self.counter % self.Xth_frame != 0:
            self.counter += 1
            return
        else:
            self.counter += 1

        try:
            # ROS CompressedImage zu OpenCV-Bild
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            bbox_array_msg = BoundingBoxArray()
            bbox_array_msg.header = image_msg.header

            if self.model is not None:
                results = self.model(cv_image, device=self.device)

                for box, conf, cls in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
                    x1, y1, x2, y2 = map(float, box)
                    class_id = int(cls)
                    confidence = float(conf)
                    label = self.model.names[class_id]

                    # Bounding Box zeichnen
                    cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(cv_image, f"{label} {confidence:.2f}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # BoundingBox Nachricht erstellen
                    bbox_msg = BoundingBox()
                    bbox_msg.header = image_msg.header
                    bbox_msg.x_min = x1
                    bbox_msg.y_min = y1
                    bbox_msg.x_max = x2
                    bbox_msg.y_max = y2
                    bbox_msg.class_id = class_id
                    bbox_msg.confidence = confidence

                    bbox_array_msg.boxes.append(bbox_msg)

                # Publish Bounding Boxes
                self.pub_boxes.publish(bbox_array_msg)

            # Fensteranzeige
            window_name = f"{self._vehicle_name} Camera + YOLO"
            cv2.imshow(window_name, cv_image)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(f"[{self.node_name}] Fehler bei Bildverarbeitung: {e}")

    def on_shutdown(self):
        cv2.destroyAllWindows()
        rospy.loginfo(f"[{self.node_name}] Geschlossene Fenster.")


if __name__ == "__main__":
    node = ShowCameraNode(node_name="show_camera_node")
    rospy.on_shutdown(node.on_shutdown)
    rospy.spin()
