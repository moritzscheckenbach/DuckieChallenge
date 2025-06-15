#!/usr/bin/env python3

import os

import rospy
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import Bool


class CameraTriggerNode(DTROS):
    def __init__(self, node_name):
        super(CameraTriggerNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle_name = os.environ["VEHICLE_NAME"]
        self.publisher = rospy.Publisher(f"/{self._vehicle_name}/capture_trigger", Bool, queue_size=10)

        self.log("Camera Trigger Node gestartet! Drücke 'p', um ein Bild aufzunehmen.")

    def run(self):
        try:
            while not rospy.is_shutdown():
                key = input("Drücke [p] für ein Foto (oder [q] zum Beenden): ")
                if key.lower() == "p":
                    msg = Bool()
                    msg.data = True
                    self.publisher.publish(msg)
                    self.log("Bildaufnahme ausgelöst")
                elif key.lower() == "q":
                    self.log("Node wird beendet")
                    rospy.signal_shutdown("User beendete die Node")
                    break
        except KeyboardInterrupt:
            self.log("Terminal wiederhergestellt.")


if __name__ == "__main__":
    node = CameraTriggerNode(node_name="camera_trigger_node")
    node.run()
