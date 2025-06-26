#!/usr/bin/env python3

import os
import random
import time
from enum import Enum

import cv2
import numpy as np
import rospkg
import rospy
import yaml
from cv_bridge import CvBridge
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

        self.config = self._load_config()
        self.crop_height_percentage = self.config["processing"]["crop_height_percentage"]  # Percentage of the image height to crop from the top

        self.bridge = CvBridge()

        # Subscribers
        self.sub_control_mode = rospy.Subscriber(f"/{self._vehicle_name}/current_mode", Int32MultiArray, self.cbControlMode, queue_size=1)
        # self.sub_lane = rospy.Subscriber(f"/{self._vehicle_name}/detect/lane", Float64, self.cblane, queue_size=1)
        self.sub_masks = rospy.Subscriber(f"/{self._vehicle_name}/detect/masks", MultiMaskGroups, self.cbmasks, queue_size=1)
        self.sub_image = rospy.Subscriber(f"/{self._vehicle_name}/camera_node/image/compressed", CompressedImage, self.cbimage, queue_size=1)
        # Publishers
        self.pub_cmd_vel = rospy.Publisher(f"/{self._vehicle_name}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)
        self.pub_intersection_done = rospy.Publisher(f"/{self._vehicle_name}/intersection_handled", Bool, queue_size=1)

        rospy.loginfo(f"{self._vehicle_name}: IntersectionHandlingNode initialized with state: {self._state}")

    def _load_config(self):
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("default")  # Name deines Packages!
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

    def cbControlMode(self, msg: Int32MultiArray):
        if msg.data[7] == 1:
            self._node_active = True
            rospy.logwarn(f"{self._vehicle_name}: IntersectionHandlingNode is now active")
            # rospy.sleep(0.5)
            self.classifyIntersectionType()
        else:
            self._node_active = False

    def cbmasks(self, msg: MultiMaskGroups):
        if not self._node_active:
            return

        try:
            self.red_masks = [self.bridge.imgmsg_to_cv2(m, desired_encoding="mono8") for m in msg.red]
            self.img = self.pre_img if hasattr(self, "pre_img") else None  # Use pre_img if available
            if not self.red_masks:
                rospy.logwarn_throttle(1.0, f"{self._vehicle_name}: No red masks received")
            else:
                self.visualizeIntersection()
        except Exception as e:
            rospy.logerr(f"{self._vehicle_name}: Error processing masks: {e}")

    def cbimage(self, msg: CompressedImage):
        try:
            # Convert compressed image to OpenCV format
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.pre_img = self.crop_img(cv_image)  # Crop the image if needed
            # Process the image if needed (e.g., visualization)
            # rospy.loginfo(f"{self._vehicle_name}: Received image")
        except Exception as e:
            rospy.logerr(f"{self._vehicle_name}: Error processing image: {e}")

    def crop_img(self, img):
        img = img.copy()
        h, w = img.shape[:2]
        crop_height = int(h * self.crop_height_percentage)  # Crop X% from the top
        img = img[crop_height:, :]
        self.image_height = img.shape[0]
        rospy.loginfo(f"image size: height:{img.shape[0]}, width:{img.shape[1]}")

        return img

        # NOTE: If needed place bird's eye view transformation here

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

            else:
                # Process each red line mask
                red_masks = self.red_masks
                for red_mask in red_masks:
                    rospy.logwarn(f"Processing red mask with shape: {red_mask.shape}")
                    # Resize red mask if necessary to match direction mask dimensions
                    # Add validation checks
                    if red_mask is not None and left_mask is not None and red_mask.size > 0 and left_mask.size > 0:
                        if red_mask.shape != left_mask.shape:
                            try:
                                red_mask = cv2.resize(red_mask, (left_mask.shape[1], left_mask.shape[0]))
                                rospy.logwarn("Resized red mask to match direction mask dimensions.")
                            except Exception as e:
                                rospy.logerr(f"Resize operation failed: {e}")
                                rospy.logerr(f"Red mask shape: {red_mask.shape}, Left mask shape: {left_mask.shape}")
                    else:
                        rospy.logerr("Cannot resize: One of the masks is None or empty")

                    # Calculate overlap with each direction
                    left_overlap = self._calculate_mask_overlap(red_mask, left_mask)
                    rospy.logwarn(f"Left overlap: {left_overlap}")
                    straight_overlap = self._calculate_mask_overlap(red_mask, straight_mask)
                    rospy.logwarn(f"Straight overlap: {straight_overlap}")
                    right_overlap = self._calculate_mask_overlap(red_mask, right_mask)
                    rospy.logwarn(f"Right overlap: {right_overlap}")

                    # Update availability based on overlap threshold (50%)
                    if left_overlap > 10:
                        left_available = True
                    if straight_overlap > 10:
                        straight_available = True
                    if right_overlap > 10:
                        right_available = True

        except Exception as e:
            rospy.logerr(f"Error processing direction masks: {e}")

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

        rospy.logwarn(f"Classified intersection type: {intersection_type}")
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
        detected_binary = np.sum(detected_binary)

        # Avoid division by zero
        if detected_binary == 0:
            return 0

        # Calculate percentage of template covered by detection
        overlap_percentage = (intersection_area / detected_binary) * 100

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
            rospy.logwarn(f"Chosen intersection direction: {self._intersection_direction}")
        else:
            self._intersection_direction = IntersectionDirection.STRAIGHT
            rospy.logwarn("No valid intersection direction found, defaulting to STRAIGHT")

        # NOTE: If traffic rules need to be checked, implement that logic here

        self.turn()

    def turn(self):
        self._state = IntersectionHandlingNodeState.EXECUTING_ACTION

        # Create message for velocity commands
        cmd_msg = Twist2DStamped()

        # Define parameters for each turn type
        straight_time = 1.2  # Time to go straight (seconds)
        straight_time_right = 0.8
        turn_time = 1.5  # Time to execute turn (seconds)
        v_straight = 0.8  # Linear velocity for straight (m/s)
        v_turn = 0.15  # Linear velocity during turn (m/s)
        omega_left = 5  # Angular velocity for left turn (rad/s)
        omega_right = -5  # Angular velocity for right turn (rad/s)

        start_time = time.time()
        rate = rospy.Rate(10)  # 10Hz control loop

        if self._intersection_direction == IntersectionDirection.LEFT:
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

        elif self._intersection_direction == IntersectionDirection.STRAIGHT:
            rospy.loginfo("Going straight")

            # Go straight for defined distance/time
            while time.time() - start_time < straight_time:
                cmd_msg.v = v_straight
                cmd_msg.omega = 0.0
                self.pub_cmd_vel.publish(cmd_msg)
                rate.sleep()

        elif self._intersection_direction == IntersectionDirection.RIGHT:
            rospy.loginfo("Turning right")

            # First go straight for a bit
            while time.time() - start_time < straight_time_right:
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

        # Stop the robot
        cmd_msg.v = 0.0
        cmd_msg.omega = 0.0
        self.pub_cmd_vel.publish(cmd_msg)

        # Mark intersection handling as completed
        self.publishDone()

    def publishDone(self):
        self._state = IntersectionHandlingNodeState.COMPLETED
        self.pub_intersection_done.publish(Bool(data=True))
        rospy.loginfo(f"{self._vehicle_name}: Intersection handling completed, state: {self._state}")

        self._node_active = False
        rospy.loginfo(f"{self._vehicle_name}: IntersectionHandlingNode is now inactive")

    def visualizeIntersection(self):
        """
        Visualize the intersection with red masks overlaid on the image.
        Red masks are overlaid with 50% transparency.
        """
        rospy.logwarn(f"{self._vehicle_name}: Visualizing intersection with {len(self.red_masks)} red masks")
        if not self.red_masks:
            rospy.logwarn_throttle(1.0, f"{self._vehicle_name}: No red masks to visualize")
            return

        try:
            # Create a blank visualization image (all black)
            # Assuming masks are same size, use the first one for dimensions
            height, width = self.red_masks[0].shape[:2]
            vis_img = self.img.copy() if hasattr(self, "img") else np.zeros((height, width, 3), dtype=np.uint8)

            # Overlay each red mask with 50% transparency
            for mask in self.red_masks:
                # Convert binary mask to color (red)
                colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
                colored_mask[mask > 0] = [0, 0, 255]  # Red in BGR

                # Overlay with 50% transparency
                vis_img = cv2.addWeighted(vis_img, 1.0, colored_mask, 0.5, 0)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(vis_img, "Intersection", (10, 30), font, 1, (255, 255, 255), 2)

                # Display the visualization
                cv2.imshow("Intersecion Classification", vis_img)
                cv2.waitKey(1)

            rospy.loginfo(f"{self._vehicle_name}: Intersection visualization created with {len(self.red_masks)} masks")
        except Exception as e:
            rospy.logerr(f"{self._vehicle_name}: Error in visualization: {e}")


if __name__ == "__main__":
    node = IntersectionHandlingNode(node_name="intersection_handling_node")
    rospy.spin()
