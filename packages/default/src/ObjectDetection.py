#!/usr/bin/env python3

import os
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from ultralytics import YOLO

from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from geometry_msgs.msg import Pose2D


class ShowCameraNode(DTROS):
    """
    DTROS-Node, das das Kamerabild abonniert, YOLO-Detections ausführt
    und die Ergebnisse als Bild sowie als Detection2DArray publiziert.
    """

    def __init__(self, node_name):
        super(ShowCameraNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.VISUALIZATION
        )

        self._vehicle_name = os.environ.get("VEHICLE_NAME", "")
        if not self._vehicle_name:
            rospy.logwarn("VEHICLE_NAME nicht gesetzt, verwende Standard-Topic")

        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self.bridge = CvBridge()

        # YOLO-Modell laden
        model_path = "/home/duckie6/ConnorMCQuackor/DuckieChallenge_git/DuckieChallenge-main/packages/duckie_detection/src/model/yolo_v8s_duckiedetection.pt"
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            rospy.loginfo(f"YOLO-Modell geladen: {model_path}")
        else:
            rospy.logerr(f"YOLO-Modell nicht gefunden: {model_path}")
            self.model = None

        # Publisher für Detections
        self.pub_all_masks = rospy.Publisher(
            f"/{self._vehicle_name}/detect/objects",
            Detection2DArray,
            queue_size=1
        )

        # Bild-Subscriber
        self.sub_image = rospy.Subscriber(
            self._camera_topic,
            CompressedImage,
            self.cb_display_image,
            queue_size=1
        )
        rospy.loginfo(f"[{self.node_name}] Abonniert: {self._camera_topic}")

    def cb_display_image(self, image_msg):
        try:
            # ROS CompressedImage zu OpenCV-Bild
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            detection_array_msg = Detection2DArray()
            detection_array_msg.header = image_msg.header

            if self.model is not None:
                results = self.model(cv_image)

                for box, conf, cls in zip(
                    results[0].boxes.xyxy,
                    results[0].boxes.conf,
                    results[0].boxes.cls
                ):
                    x1, y1, x2, y2 = map(int, box)
                    label = self.model.names[int(cls)]
                    score = float(conf)

                    # Bounding Box zeichnen
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        cv_image,
                        f"{label} {score:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1
                    )

                    # ROS Detection2D Nachricht bauen
                    detection = Detection2D()
                    detection.bbox.center.x = (x1 + x2) / 2.0
                    detection.bbox.center.y = (y1 + y2) / 2.0
                    detection.bbox.size_x = x2 - x1
                    detection.bbox.size_y = y2 - y1

                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.id = int(cls)  # numerische Klasse
                    hypothesis.score = score

                    detection.results.append(hypothesis)
                    detection_array_msg.detections.append(detection)

                # Publish
                self.pub_all_masks.publish(detection_array_msg)

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
