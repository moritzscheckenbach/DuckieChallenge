#!/usr/bin/env python3

import os
from collections import defaultdict
from enum import Enum
from typing import List

import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import Range
from std_msgs.msg import Bool, Int32MultiArray, String

NODE_INDEX = {
    "NormalLaneFollowing": 0,
    "DuckieCheckCenter": 1,
    "OppositeLaneFollowing": 2,
    "DuckieCheckRight": 3,
    "PIDControlLane": 4,
    "IntersectionDetection": 5,
    "StopAtIntersection": 6,
    "IntersectionHandling": 7,
    "SearchForParkingLot": 8,
    "StopAtParkingLot": 9,
    "StopVehicle": 10,
    "ParkStopGo": 11,
    "RedMaskDetection": 12,
}


def create_bitvector(active_nodes: List[str], length: int = len(NODE_INDEX)) -> List[int]:
    vec = [0] * length
    for name in active_nodes:
        idx = NODE_INDEX[name]
        vec[idx] = 1
    print(f"Created bitvector for {active_nodes}: {vec}")
    return vec


class ControlMode(Enum):

    NormalLaneFollowing = create_bitvector(
        [
            "NormalLaneFollowing",
            "DuckieCheckCenter",
            "PIDControlLane",
            "RedMaskDetection",
            "SearchForParkingLot",
        ]
    )

    IntersectionProximity = create_bitvector(
        [
            "NormalLaneFollowing",
            "PIDControlLane",
            "IntersectionDetection",
            "RedMaskDetection",
        ]
    )

    AvoidDuckies = create_bitvector(
        [
            "OppositeLaneFollowing",
            "DuckieCheckRight",
            "PIDControlLane",
        ]
    )

    StoppingAtIntersection = create_bitvector(
        [
            "StopAtIntersection",
        ]
    )

    IntersectionHandling = create_bitvector(
        [
            "IntersectionHandling",
        ]
    )

    SearchForParkingLot = create_bitvector(
        [
            "NormalLaneFollowing",
            "PIDControlLane",
            "StopAtParkingLot",
        ]
    )

    StopAtParkingLot = create_bitvector(
        [
            "NormalLaneFollowing",
            "PIDControlLane",
            "StopAtParkingLot",
        ]
    )

    StoppedAtParkingLot = create_bitvector(
        [
            "StopVehicle",
        ]
    )

    ParkingManager = create_bitvector(
        [
            "ParkStopGo",
        ]
    )

    EmergencyStop = create_bitvector(
        [
            "StopVehicle",
        ]
    )


class ControlMode_old(Enum):

    NormalLaneFollowing = [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0]
    AvoidDuckies = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    StoppingAtIntersection = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    IntersectionHandling = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    SearchForParkingLot = [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
    StopAtParkingLot = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
    ParkingManager = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]


class AdminNode(DTROS):
    def __init__(self, node_name):
        super(AdminNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle_name = os.environ["VEHICLE_NAME"]

        rospy.Subscriber(f"/{self._vehicle_name}/front_center_tof_driver_node/range", Range, self._on_range_sensor_data, queue_size=1)
        rospy.Subscriber(f"/{self._vehicle_name}/detect/in_region", Bool, self._on_duckie_detected, queue_size=1)
        rospy.Subscriber(f"/{self._vehicle_name}/redstop_detected", Bool, self._on_redstop_detected, queue_size=1)
        rospy.Subscriber(f"/{self._vehicle_name}/parkinglot_detected", Bool, self._on_parkinglot_detected, queue_size=1)

        rospy.Subscriber(f"/{self._vehicle_name}/detect/not_in_region", Bool, self._back_to_lane_following, queue_size=1)
        rospy.Subscriber(f"/{self._vehicle_name}/intersection_handled", Bool, self._back_to_lane_following, queue_size=1)
        rospy.Subscriber(f"/{self._vehicle_name}/duckiebot_parked", Bool, self._back_to_lane_following, queue_size=1)
        rospy.Subscriber(f"/{self._vehicle_name}/vehicle_stopped", Bool, self._go_to_intersection_handling, queue_size=1)
        rospy.Subscriber(f"/{self._vehicle_name}/parking/possible", Bool, self.go_to_stop_at_parking_lot, queue_size=1)
        rospy.Subscriber(f"/{self._vehicle_name}/stopped_at_parkinglot", Bool, self.go_to_parking_manager, queue_size=1)
        rospy.Subscriber(f"/{self._vehicle_name}/parking/occupied", Bool, self._set_occupied_status, queue_size=1)
        rospy.Subscriber(f"/{self._vehicle_name}/stopp_command", Bool, self._go_to_parkinglot_stopp, queue_size=1)
        rospy.Subscriber(f"/{self._vehicle_name}/redmask_detected", Bool, self._go_to_proximity_handling, queue_size=1)

        self.current_mode = ControlMode.NormalLaneFollowing
        self.status_pub = rospy.Publisher(f"/{self._vehicle_name}/current_mode", Int32MultiArray, queue_size=1, latch=True)
        self._publish_mode()

        self.occupied_status = False

        rospy.logwarn(f"[AdminNode] Initialisiert. Aktueller Modus: {self.current_mode.name}")

    def _on_duckie_detected(self, msg):
        if msg.data and self.current_mode == ControlMode.NormalLaneFollowing:
            rospy.logwarn("Duckie erkannt! Wechsel in 'AvoidDuckies'-Modus")
            self.current_mode = ControlMode.AvoidDuckies
            self._publish_mode()

    def _on_redstop_detected(self, msg):
        if msg.data and self.current_mode == ControlMode.IntersectionProximity:
            rospy.logwarn("Redstop erkannt! Wechsel in 'StoppingAtIntersection'-Modus")
            self.current_mode = ControlMode.StoppingAtIntersection
            self._publish_mode()

    def _go_to_intersection_handling(self, msg):
        if msg.data and self.current_mode == ControlMode.StoppingAtIntersection:
            rospy.logwarn("Fahrzeug gestoppt! Wechsel in 'IntersectionHandling'-Modus")
            self.current_mode = ControlMode.IntersectionHandling
            self._publish_mode()

    def _on_parkinglot_detected(self, msg):
        if msg.data and self.current_mode == ControlMode.NormalLaneFollowing:
            rospy.logwarn("Parkplatz erkannt! Wechsel in 'SearchForParkingLot'-Modus")
            self.current_mode = ControlMode.SearchForParkingLot
            self._publish_mode()

    def go_to_stop_at_parking_lot(self, msg):
        # if msg.data and self.current_mode == ControlMode.SearchForParkingLot:
        if msg.data and self.current_mode == ControlMode.NormalLaneFollowing:
            rospy.logwarn("Parkplatz gefunden! Wechsel in 'StopAtParkingLot'-Modus")
            self.current_mode = ControlMode.StopAtParkingLot
            self._publish_mode()

    def _go_to_parkinglot_stopp(self, msg):
        if msg.data and self.current_mode == ControlMode.StopAtParkingLot:
            rospy.logwarn("Parkplatz gestoppt! Wechsel in 'StoppedAtParkingLot'-Modus")
            self.current_mode = ControlMode.StoppedAtParkingLot
            self._publish_mode()

    def go_to_parking_manager(self, msg):
        if msg.data and self.current_mode == ControlMode.StoppedAtParkingLot and self.occupied_status == False:
            rospy.logwarn("Am Parkplatz gestoppt! Wechsel in 'ParkingManager'-Modus")
            self.current_mode = ControlMode.ParkingManager
            self._publish_mode()
        else:
            rospy.logwarn("Zurück zum 'NormalLaneFollowing'-Modus (Parkplatz belegt oder nicht gestoppt)")
            self.current_mode = ControlMode.NormalLaneFollowing
            self._publish_mode()

    def _back_to_lane_following(self, msg):
        if msg.data and self.current_mode != ControlMode.NormalLaneFollowing:
            rospy.logwarn("Zurück zum 'NormalLaneFollowing'-Modus")
            self.current_mode = ControlMode.NormalLaneFollowing
            self._publish_mode()

    def _on_range_sensor_data(self, msg):
        if msg.range < 0.2 and self.current_mode != ControlMode.EmergencyStop:
            last_mode = self.current_mode
            rospy.logwarn("Notbremsung ausgelöst! Wechsel in 'EmergencyStop'-Modus")
            self.current_mode = ControlMode.EmergencyStop
            self._publish_mode()
        elif msg.range >= 0.2 and self.current_mode == ControlMode.EmergencyStop:
            rospy.logwarn("Notbremsung aufgehoben! Zurück zum vorherigen Modus")
            self.current_mode = last_mode if "last_mode" in locals() else ControlMode.NormalLaneFollowing
            rospy.sleep(0.5)
            self._publish_mode()
        elif msg.range >= 0.2 and self.current_mode != ControlMode.EmergencyStop:
            pass

    def _go_to_proximity_handling(self, msg):
        if msg.data and self.current_mode == ControlMode.NormalLaneFollowing:
            rospy.logwarn("Redmask erkannt! Wechsel in 'IntersectionProximity'-Modus")
            self.current_mode = ControlMode.IntersectionProximity
            self._publish_mode()
        else:
            rospy.logwarn("Redstop nicht erkannt oder im falschen Modus. Zurück zum 'NormalLaneFollowing'-Modus")
            self.current_mode = ControlMode.NormalLaneFollowing
            self._publish_mode()

    def _set_occupied_status(self, msg):
        self.occupied_status = msg.data

    def _publish_mode(self):
        msg = Int32MultiArray()
        msg.data = self.current_mode.value
        self.status_pub.publish(msg)


if __name__ == "__main__":
    node = AdminNode(node_name="AdminNode")
    rospy.spin()
