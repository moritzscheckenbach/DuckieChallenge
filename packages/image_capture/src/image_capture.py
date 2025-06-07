import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool


class ImageCaptureNode(Node):
    def __init__(self):
        super().__init__("image_capture_node")

        self.bridge = CvBridge()

        # Subscriber für Bilder
        self.image_subscription = self.create_subscription(Image, "/camera/camera/color/image_raw", self.image_callback, 10)

        # Subscriber für den Trigger
        self.trigger_subscription = self.create_subscription(Bool, "/capture_trigger", self.trigger_callback, 10)

        self.image_msg = None
        self.image_counter = 0

    def image_callback(self, msg):
        """Speichert das aktuelle Bild zwischen."""
        self.image_msg = msg

    def trigger_callback(self, msg):
        """Reagiert auf den Trigger und speichert ein Bild."""
        if self.image_msg == None:
            self.get_logger().info("Noch kein Bild empfangen, Trigger wird ignoriert!")
            return

        self.get_logger().info(f"Bild {self.image_counter +1} empfangen, Anzeige wird geöffnet")

        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.image_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().info(f"Fehler bei der Bildkonvertierung, bitte versuchen Sie es erneut")
        # save_path = (f"extrinsic_camera_calibration/extrinsic_camera_calibration/calibration_images/captured_image_{self.image_counter + 1:03}.jpg")
        save_path = f"src/extrinsic_camera_calibration/extrinsic_camera_calibration/calibration_data/images/{self.image_counter + 1:03}.jpg"
        save_complete = cv2.imwrite(save_path, cv_image)
        if save_complete:
            self.get_logger().info(f"Bild gespeichert: {save_path}")
            self.image_counter += 1
        else:
            self.get_logger().info(f"Fehler beim Speichern des Bildes {self.image_counter +1}!")


def main(args=None):
    rclpy.init(args=args)
    node = ImageCaptureNode()
    rclpy.spin(node)
    rclpy.shutdown()
