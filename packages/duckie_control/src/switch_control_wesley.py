#!/usr/bin/env python3

import os
from collections import defaultdict

import rospy
from std_msgs.msg import Bool, String

# Dummy: Liste aller Nodes, die existieren (hier nur symbolisch; bitte ersetzen)
ALL_NODES = ["LaneSegmentation", "LaneFollowing", "ObstacleAvoidance", "IntersectionManager", "DuckieDetection"]

# Definierte Systemmodi und die jeweils aktiven Nodes
MODE_CONFIG = {
    "normal_drive": ["LaneSegmentation", "LaneFollowing", "ObjectDetection"],
    "avoid_obstacle": ["LaneSegmentation", "ObjectDetection", "OppositeLaneFollowing"],
    "avoid_duckiebot": ["LaneSegmentation", "ObjectDetection"],
    "intersection_handling": ["LaneSegmentation", "ObjectDetection", "IntersectionHandling"],
    "parking": ["LaneSegmentation", "ObjectDetection", "ParkingManager"],
}

# Hier kann auf Events gehört werden (z. B. Duckie erkannt)
EVENT_TOPIC = "/duckie_detected"


class AdminStatus:
    """
    Einfache Datenstruktur, simuliert eine benutzerdefinierte Message
    """

    def __init__(self):
        self.node_status = defaultdict(bool)  # z. B. {"LaneSegmentation": True, "...": False}


class AdminNode:
    def __init__(self):
        rospy.init_node("admin_node")

        self._vehicle_name = os.environ["VEHICLE_NAME"]

        self.current_mode = "normal_drive"
        self.status_pub = rospy.Publisher(f"/{self._vehicle_name}/current_mode", String, queue_size=1, latch=True)
        # self.bool_pub = rospy.Publisher(f"/{self._vehicle_name}/admin_node_enable_map", Bool, queue_size=1, latch=True)

        # Placeholder für Bool-Map pro Node
        self.node_map_pub = rospy.Publisher(f"/{self._vehicle_name}/admin_node_states", String, queue_size=1, latch=True)

        # Sub auf Event, z. B. Duckie erkannt
        rospy.Subscriber(f"/{self._vehicle_name}/duckiebot_detected", Bool, self._on_duckiebot_detected)
        rospy.Subscriber(f"/{self._vehicle_name}/redstop_detected", Bool, self._on_duckiebot_detected)

        rospy.Subscriber(f"/{self._vehicle_name}/duckiebot_avoided", Bool, self._back_to_lane_following)
        rospy.Subscriber(f"/{self._vehicle_name}/ducie_avoided", Bool, self._back_to_lane_following)
        rospy.Subscriber(f"/{self._vehicle_name}/intersection_handled", Bool, self._back_to_lane_following)
        rospy.Subscriber(f"/{self._vehicle_name}/duckiebot_parked", Bool, self._back_to_lane_following)

        # Regelmäßige FSM-Ausführung
        self.timer = rospy.Timer(rospy.Duration(1.0), self._publish_status)

    def _on_duckie_detected(self, msg):
        if msg.data and self.current_mode != "avoid_obstacle":
            rospy.loginfo("Duckie erkannt! Wechsel in 'avoid_obstacle'-Modus")
            self.current_mode = "avoid_obstacle"

    def _on_duckiebot_detected(self, msg):
        if msg.data and self.current_mode != "idle":
            rospy.loginfo("Duckiebot erkannt! Wechsel in 'idle'-Modus")
            self.current_mode = "duckiebot_handling"

    def _on_parking_lot_detected(self, msg):
        if msg.data and self.current_mode != "idle":
            rospy.loginfo("Parkplatz erkannt! Wechsel in 'idle'-Modus")
            self.current_mode = "parking"

    def _on_intersection_detected(self, msg):
        if msg.data and self.current_mode != "idle":
            rospy.loginfo("Kreuzung erkannt! Wechsel in 'idle'-Modus")
            self.current_mode = "intersection_handling"

    def _back_to_lane_following(self, msg):
        if self.current_mode != "normal_drive" and msg.data == True:
            rospy.loginfo("Zurück zum 'normal_drive'-Modus")
            self.current_mode = "normal_drive"

    def _publish_status(self, event):
        active_nodes = MODE_CONFIG.get(self.current_mode, [])
        rospy.loginfo(f"[AdminNode] Modus: {self.current_mode}, Aktive Nodes: {active_nodes}")

        # Veröffentliche simplen String-Modus
        self.status_pub.publish(self.current_mode)

        # Optional: Veröffentlichung als bool-Map pro Node
        node_status_str = "".join([f"{name}:{str(name in active_nodes).lower()}," for name in ALL_NODES])
        self.node_map_pub.publish(node_status_str)


if __name__ == "__main__":
    try:
        AdminNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
