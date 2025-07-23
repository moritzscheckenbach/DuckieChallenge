#!/usr/bin/env python3

import os

import numpy as np
import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from normal_lane_following.msg import MultiMaskGroups
from std_msgs.msg import Bool, Float64, Int32MultiArray, String


class StopAtParkingLot(DTROS):
    def __init__(self, node_name):
        super(StopAtParkingLot, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle_name = os.environ.get("VEHICLE_NAME", "")
        self._node_active = False
        self.bridge = CvBridge()

        # Timer-System für robuste Erkennung
        self.dotted_timeout = 0.45  # 0.3 Sekunden Timeout
        self.last_dotted_seen = rospy.Time.now()
        self.dotted_missing_published = False  # Verhindert mehrfaches Publizieren

        self.sub_control_mode = rospy.Subscriber(f"/{self._vehicle_name}/current_mode", Int32MultiArray, self.cbControlMode, queue_size=1)
        self.sub_masks = rospy.Subscriber(f"/{self._vehicle_name}/detect/masks", MultiMaskGroups, self.mask_callback, queue_size=1)
        self.pub_halt = rospy.Publisher(f"/{self._vehicle_name}/stopp_command", Bool, queue_size=1)

        rospy.loginfo("[DottedMaskChecker] Node gestartet")

    def cbControlMode(self, msg: Int32MultiArray):
        if msg.data[9] == 1:
            self._node_active = True
            # Reset Timer beim Aktivieren
            self.last_dotted_seen = rospy.Time.now()
            self.dotted_missing_published = False
            rospy.loginfo(f"{self._vehicle_name}: Parking check active")
        else:
            self._node_active = False

    def mask_callback(self, msg: MultiMaskGroups):
        if not self._node_active:
            return

        current_time = rospy.Time.now()

        # Konvertiere dotted-Masken
        dotted_masks = [self.bridge.imgmsg_to_cv2(m, desired_encoding="mono8") for m in msg.dotted]

        # Prüfe ob dotted_mask erkannt wurde
        dotted_detected = len(dotted_masks) > 0

        if dotted_detected:
            # Dotted Linie gesehen - Timer zurücksetzen
            self.last_dotted_seen = current_time
            self.dotted_missing_published = False
            rospy.loginfo_throttle(1.0, "[DottedMaskChecker] Dotted line detected - timer reset")
        else:
            # Keine dotted Linie - prüfe Timer
            time_since_last_seen = (current_time - self.last_dotted_seen).to_sec()

            if time_since_last_seen >= self.dotted_timeout:
                # Timer abgelaufen - dotted missing
                if not self.dotted_missing_published:
                    self.pub_halt.publish(Bool(True))
                    self.dotted_missing_published = True
                    rospy.logwarn(f"[DottedMaskChecker] Dotted missing for {time_since_last_seen:.2f}s - STOPPING!")
            else:
                # Timer läuft noch - warten
                remaining_time = self.dotted_timeout - time_since_last_seen
                rospy.loginfo_throttle(0.1, f"[DottedMaskChecker] Dotted missing - waiting {remaining_time:.2f}s")


if __name__ == "__main__":
    node = StopAtParkingLot(node_name="stop_if_no_dotted_mask_node")
    rospy.spin()
