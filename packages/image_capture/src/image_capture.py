#!/usr/bin/env python3

import os

import cv2
import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool


class ImageCaptureNode(DTROS):
    def __init__(self, node_name):
        super(ImageCaptureNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self.bridge = CvBridge()

        # Bestimme Vehicle-Namen für die Topics
        self._vehicle_name = os.environ["VEHICLE_NAME"]

        # Subscriber für Bilder
        self.image_subscription = rospy.Subscriber(f"/{self._vehicle_name}/camera_node/image/compressed", Image, self.image_callback, queue_size=1)

        # Subscriber für den Trigger
        self.trigger_subscription = rospy.Subscriber(f"/{self._vehicle_name}/capture_trigger", Bool, self.trigger_callback, queue_size=1)

        self.image_msg = None
        self.image_counter = 0

        # Erstelle den Speicherordner, falls er nicht existiert
        self.save_dir = os.path.join(os.environ.get("DT_REPO_PATH", "/code"), "packages", "image_capture", "images")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.log("Image Capture Node gestartet!")

    def image_callback(self, msg):
        """Speichert das aktuelle Bild zwischen."""
        self.image_msg = msg

    def trigger_callback(self, msg):
        """Reagiert auf den Trigger und speichert ein Bild."""
        if self.image_msg is None:
            self.log("Noch kein Bild empfangen, Trigger wird ignoriert!", "warn")
            return

        self.log(f"Bild {self.image_counter + 1} empfangen, wird gespeichert...")

        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(self.image_msg)
            save_path = os.path.join(self.save_dir, f"captured_image_{self.image_counter + 1:03}.jpg")
            save_complete = cv2.imwrite(save_path, cv_image)

            if save_complete:
                self.log(f"Bild gespeichert: {save_path}")
                self.image_counter += 1
            else:
                self.log(f"Fehler beim Speichern des Bildes {self.image_counter + 1}!", "warn")

        except Exception as e:
            self.log(f"Fehler bei der Bildkonvertierung: {str(e)}", "error")


if __name__ == "__main__":
    node = ImageCaptureNode(node_name="image_capture_node")
    rospy.spin()
