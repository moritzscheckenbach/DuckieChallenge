#!/usr/bin/env python3

import os

import rospy
import yaml
from std_msgs.msg import Bool, Float64, Int32, Int32MultiArray, String
from vision_msgs.msg import Detection2DArray


class DetectionCheckerNode:
    def __init__(self):
        rospy.init_node("detection_checker_node")

        # Konfiguration laden
        config_path = rospy.get_param("~config_path")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.target_class_id = config["target_class_id"]
        self.region = config["region"]

        vehicle_name = os.environ.get("VEHICLE_NAME", "")

        self._node_active = False

        self.sub = rospy.Subscriber(f"/{vehicle_name}/detect/objects", Detection2DArray, self.detection_callback, queue_size=1)
        self.sub_control_mode = rospy.Subscriber(f"/{self._vehicle_name}/current_mode", Int32MultiArray, self.cbControlMode, queue_size=1)

        self.pub_in_region = rospy.Publisher(f"/{vehicle_name}/detect/in_region", Bool, queue_size=1)

        rospy.loginfo(f"[DetectionCheckerNode] Läuft. Überwacht Klasse {self.target_class_id} im Bereich {self.region}")

    def cbControlMode(self, msg: Int32MultiArray):
        if msg.data[1] == 1:
            self._node_active = True
            rospy.loginfo(f"{self._vehicle_name}: DuckieCenterCheck is now active")
        else:
            self._node_active = False

    def detection_callback(self, msg):
        if self._node_active == True:
            in_region = False

            for detection in msg.detections:
                for hypothesis in detection.results:
                    if hypothesis.id == self.target_class_id:
                        x = detection.bbox.center.x
                        y = detection.bbox.center.y

                        if self.region["x_min"] <= x <= self.region["x_max"] and self.region["y_min"] <= y <= self.region["y_max"]:
                            in_region = True
                            break

            self.pub_in_region.publish(Bool(data=in_region))


if __name__ == "__main__":
    try:
        DetectionCheckerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
