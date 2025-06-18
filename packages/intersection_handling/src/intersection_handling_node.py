#!/usr/bin/env python3

import os
import random
import time
from enum import Enum

import cv2
import numpy as np
import rospkg
import rospy
from cv_bridge import CvBridge
from duckie_control.src.switch_control_node import ControlType
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from normal_lane_following.msg import MultiMaskGroups
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Float64, Int32, Int32MultiArray, String


class IntersectionHandlingNodeState(Enum):
    OFF = "off"
    CLASSIFYING_INTERSECTION = "classifying_intersection"
    CHOOSING_DIRECTION = "choosing_direction"
    CHECK_TRAFFIC_RULES = "checking_traffic_rules"
    EXECUTING_ACTION = "executing_action"
    COMPLETED = "completed"


class IntersectionDirection(Enum):
    LEFT = "left"
    STRAIGHT = "straight"
    RIGHT = "right"


class IntersectionHandlingNode(DTROS):
    def __init__(self, node_name):
        super(IntersectionHandlingNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle_name = os.environ["VEHICLE_NAME"]

        # State initialization
        self._node_active = False
        self._state = IntersectionHandlingNodeState.OFF
        self._intersection_type = None
        self._intersection_direction = None
        self.red_masks = []

        self.bridge = CvBridge()

        # Subscribers
        self.sub_control_mode = rospy.Subscriber(f"/{self._vehicle_name}/current_mode", Int32MultiArray, self.cbControlMode, queue_size=1)
        # self.sub_lane = rospy.Subscriber(f"/{self._vehicle_name}/detect/lane", Float64, self.cblane, queue_size=1)
        self.sub_masks = rospy.Subscriber(f"/{self._vehicle_name}/detect/masks", String, self.cbmasks, queue_size=1)

        # Publishers
        self.pub_cmd_vel = rospy.Publisher(f"/{self._vehicle_name}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)
        self.pub_intersection_done = rospy.Publisher(f"/{self._vehicle_name}/intersection/done", Bool, queue_size=1)

        rospy.loginfo(f"{self._vehicle_name}: IntersectionHandlingNode initialized with state: {self._state}")

    def cbControlMode(self, msg: Int32MultiArray):
        if msg.data[7] == 1:
            self._node_active = True
            rospy.loginfo(f"{self._vehicle_name}: IntersectionHandlingNode is now active")
            self.classifyIntersectionType()
        else:
            self._node_active = False

    def cbmasks(self, msg: MultiMaskGroups):
        if self._node_active == True:
            self.red_masks = [self.bridge.imgmsg_to_cv2(m, desired_encoding="mono8") for m in msg.red]

    def classifyIntersectionType(self):
        self._state = IntersectionHandlingNodeState.CLASSIFYING_INTERSECTION

        # Initialize intersection directions
        left_available = False
        straight_available = False
        right_available = False

        # Path to the direction masks
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("intersection_handling")
        mask_dir = os.path.join(package_path, "direction_masks")
        left_mask_path = os.path.join(mask_dir, "c_left_exists_mask.png")
        straight_mask_path = os.path.join(mask_dir, "c_straight_exists_mask.png")
        right_mask_path = os.path.join(mask_dir, "c_right_exists_mask.png")

        # Load direction masks if they exist
        try:
            left_mask = cv2.imread(left_mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(left_mask_path) else None
            straight_mask = cv2.imread(straight_mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(straight_mask_path) else None
            right_mask = cv2.imread(right_mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(right_mask_path) else None

            # Check if we have valid masks
            if left_mask is None or straight_mask is None or right_mask is None:
                rospy.logwarn(f"One or more direction masks not found. Using default behavior.")
                # Default to using all directions
                left_available = False
                straight_available = False
                right_available = False
            else:
                # Process each red line mask
                for red_mask in self.red_masks:
                    # Resize red mask if necessary to match direction mask dimensions
                    if red_mask.shape != left_mask.shape:
                        red_mask = cv2.resize(red_mask, (left_mask.shape[1], left_mask.shape[0]))
                        rospy.logwarn("Resized red mask to match direction mask dimensions.")

                    # Calculate overlap with each direction
                    left_overlap = self._calculate_mask_overlap(red_mask, left_mask)
                    straight_overlap = self._calculate_mask_overlap(red_mask, straight_mask)
                    right_overlap = self._calculate_mask_overlap(red_mask, right_mask)

                    # Update availability based on overlap threshold (50%)
                    if left_overlap > 50:
                        left_available = True
                    if straight_overlap > 50:
                        straight_available = True
                    if right_overlap > 50:
                        right_available = True

        except Exception as e:
            rospy.logerr(f"Error processing direction masks: {e}")
            # Default to using all directions
            left_available = False
            straight_available = False
            right_available = False

        # Determine intersection type based on available directions
        intersection_type = None
        if left_available and straight_available and right_available:
            intersection_type = "LeftStraightRight"
        elif left_available and straight_available:
            intersection_type = "LeftStraight"
        elif left_available and right_available:
            intersection_type = "LeftRight"
        elif straight_available and right_available:
            intersection_type = "StraightRight"
        elif left_available:
            intersection_type = "Left"
        elif straight_available:
            intersection_type = "Straight"
        elif right_available:
            intersection_type = "Right"
        else:
            # Default if no direction is determined
            intersection_type = "Straight"
            rospy.logwarn("No valid intersection directions detected. Defaulting to Straight.")

        rospy.loginfo(f"Classified intersection type: {intersection_type}")
        self._intersection_type = intersection_type

        # Proceed to choose a direction based on intersection type
        self.chooseIntersectionDirection()

    def _calculate_mask_overlap(self, detected_mask, template_mask):
        """
        Calculate the percentage of overlap between detected mask and template mask

        Args:
            detected_mask: The detected red line mask
            template_mask: The direction template mask

        Returns:
            float: Percentage of overlap (0-100)
        """
        # Ensure both masks are binary
        detected_binary = (detected_mask > 0).astype(np.uint8)
        template_binary = (template_mask > 0).astype(np.uint8)

        # Calculate intersection and template area
        intersection = cv2.bitwise_and(detected_binary, template_binary)
        intersection_area = np.sum(intersection)
        template_area = np.sum(template_binary)

        # Avoid division by zero
        if template_area == 0:
            return 0

        # Calculate percentage of template covered by detection
        overlap_percentage = (intersection_area / template_area) * 100

        return overlap_percentage

    def chooseIntersectionDirection(self):
        self._state = IntersectionHandlingNodeState.CHOOSING_DIRECTION

        directions = []

        # Choose available directions based on intersection type
        if self._intersection_type == "LeftStraightRight":
            directions = [IntersectionDirection.LEFT, IntersectionDirection.STRAIGHT, IntersectionDirection.RIGHT]
        elif self._intersection_type == "LeftStraight":
            directions = [IntersectionDirection.LEFT, IntersectionDirection.STRAIGHT]
        elif self._intersection_type == "LeftRight":
            directions = [IntersectionDirection.LEFT, IntersectionDirection.RIGHT]
        elif self._intersection_type == "StraightRight":
            directions = [IntersectionDirection.STRAIGHT, IntersectionDirection.RIGHT]
        elif self._intersection_type == "Left":
            directions = [IntersectionDirection.LEFT]
        elif self._intersection_type == "Straight":
            directions = [IntersectionDirection.STRAIGHT]
        elif self._intersection_type == "Right":
            directions = [IntersectionDirection.RIGHT]

        if directions:
            self._intersection_direction = random.choice(directions)
            rospy.loginfo(f"Chosen intersection direction: {self._intersection_direction}")
        else:
            self._intersection_direction = IntersectionDirection.STRAIGHT
            rospy.logwarn("No valid intersection direction found, defaulting to STRAIGHT")

        # NOTE: If traffic rules need to be checked, implement that logic here
        self.turn()

    def turn(self):
        self._state = IntersectionHandlingNodeState.EXECUTING_ACTION

        match self._intersection_direction:
            case IntersectionDirection.LEFT:
                rospy.loginfo("Turning left")
                # TODO: Go straight for a bit, then turn left
            case IntersectionDirection.STRAIGHT:
                rospy.loginfo("Going straight")
                # TODO: Go straight for a defined distance
            case IntersectionDirection.RIGHT:
                rospy.loginfo("Turning right")
                # TODO: Go straight for a bit, then turn right

        twist = Twist2DStamped(v=v, omega=-pid_output)
        rospy.logwarn(f"moving {v} with omega {-pid_output} at error {current_error}")
        self.pub_cmd_vel.publish(twist)

    def turn(self):
        self._state = IntersectionHandlingNodeState.EXECUTING_ACTION

        # Create message for velocity commands
        cmd_msg = Twist2DStamped()

        # Define parameters for each turn type
        straight_time = 3.0  # Time to go straight (seconds)
        turn_time = 2.0  # Time to execute turn (seconds)
        v_straight = 0.3  # Linear velocity for straight (m/s)
        v_turn = 0.2  # Linear velocity during turn (m/s)
        omega_left = 1.0  # Angular velocity for left turn (rad/s)
        omega_right = -1.0  # Angular velocity for right turn (rad/s)

        start_time = time.time()
        rate = rospy.Rate(10)  # 10Hz control loop

        match self._intersection_direction:
            case IntersectionDirection.LEFT:
                rospy.loginfo("Turning left")

                # First go straight for a bit
                while time.time() - start_time < straight_time / 2:
                    cmd_msg.v = v_straight
                    cmd_msg.omega = 0.0
                    self.pub_cmd_vel.publish(cmd_msg)
                    rate.sleep()

                # Then execute left turn
                turn_start = time.time()
                while time.time() - turn_start < turn_time:
                    cmd_msg.v = v_turn
                    cmd_msg.omega = omega_left
                    self.pub_cmd_vel.publish(cmd_msg)
                    rate.sleep()

                # Finally go straight again
                straight_start = time.time()
                while time.time() - straight_start < straight_time / 2:
                    cmd_msg.v = v_straight
                    cmd_msg.omega = 0.0
                    self.pub_cmd_vel.publish(cmd_msg)
                    rate.sleep()

            case IntersectionDirection.STRAIGHT:
                rospy.loginfo("Going straight")

                # Go straight for defined distance/time
                while time.time() - start_time < straight_time:
                    cmd_msg.v = v_straight
                    cmd_msg.omega = 0.0
                    self.pub_cmd_vel.publish(cmd_msg)
                    rate.sleep()

            case IntersectionDirection.RIGHT:
                rospy.loginfo("Turning right")

                # First go straight for a bit
                while time.time() - start_time < straight_time / 2:
                    cmd_msg.v = v_straight
                    cmd_msg.omega = 0.0
                    self.pub_cmd_vel.publish(cmd_msg)
                    rate.sleep()

                # Then execute right turn
                turn_start = time.time()
                while time.time() - turn_start < turn_time:
                    cmd_msg.v = v_turn
                    cmd_msg.omega = omega_right
                    self.pub_cmd_vel.publish(cmd_msg)
                    rate.sleep()

                # Finally go straight again
                straight_start = time.time()
                while time.time() - straight_start < straight_time / 2:
                    cmd_msg.v = v_straight
                    cmd_msg.omega = 0.0
                    self.pub_cmd_vel.publish(cmd_msg)
                    rate.sleep()

        # # Stop the robot
        # cmd_msg.v = 0.0
        # cmd_msg.omega = 0.0
        # self.pub_cmd_vel.publish(cmd_msg)

        # Mark intersection handling as completed
        self.publishDone()

    def publishDone(self):
        self._state = IntersectionHandlingNodeState.COMPLETED
        self.pub_intersection_done.publish(Bool(data=True))
        rospy.loginfo(f"{self._vehicle_name}: Intersection handling completed, state: {self._state}")

        self._node_active = False
        rospy.loginfo(f"{self._vehicle_name}: IntersectionHandlingNode is now inactive")


if __name__ == "__main__":
    node = IntersectionHandlingNode(node_name="intersection_handling_node")
    rospy.spin()
