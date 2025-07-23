#!/usr/bin/env python3

import os

import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import LEDPattern
from std_msgs.msg import ColorRGBA, Header, String


class LEDBlinkerNode(DTROS):
    def __init__(self, node_name):
        super(LEDBlinkerNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle_name = os.environ.get("VEHICLE_NAME", "dorette")

        # LED pattern publisher
        self.pub_led_pattern = rospy.Publisher(f"/{self._vehicle_name}/led_emitter_node/led_pattern", LEDPattern, queue_size=1)

        # Subscriber for blinker commands
        self.sub_blinker_cmd = rospy.Subscriber(f"/{self._vehicle_name}/blinker_command", String, self.blinker_command_callback, queue_size=1)

        # Blink parameters
        self.blink_frequency = 2.0  # 2 Hz = 0.5 seconds on/off
        self.is_on = True
        self.current_mode = "off"  # "off", "left", "right"

        # Timer for blinking
        self.blink_timer = None

        rospy.loginfo(f"{self._vehicle_name}: LED Blinker Node initialized")
        rospy.loginfo(f"Subscribe to: /{self._vehicle_name}/blinker_command with 'off', 'left', or 'right'")

    def create_led_pattern(self, mode="off", led_state=True):
        """
        Create LED pattern message for turn signals

        Args:
            mode (str): "off", "left", "right", or "straight"
            led_state (bool): True for on, False for off (for blinking)
        """
        msg = LEDPattern()

        # Header
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = ""

        # LED indices mapping - we need to test which LED is which
        # Based on your feedback: currently front_right shows red (wrong), should be white
        # back LEDs show white (wrong), should be red
        # This suggests the LED mapping or color assignment is wrong

        # Let's implement a more systematic approach:
        # Real vehicle standard: Front lights = white (headlights), Rear lights = red (taillights)
        rgb_vals = []
        color_list = []

        for i in range(5):
            color = ColorRGBA()
            color_name = "off"

            # Let's try a different mapping based on your observations:
            # If front_right (index 1) shows red, and back LEDs show white,
            # then maybe the indices don't match our assumption

            # New approach: define colors by index based on observed behavior
            if i == 0:  # front_left - observed as white (correct)
                default_r, default_g, default_b = 1.0, 1.0, 1.0
                default_color_name = "white"
            elif i == 1:  # back_right - observed as red (should be white)
                default_r, default_g, default_b = 1.0, 0.0, 0.0  # Force to white
                default_color_name = "red"
            elif i == 2:  # front_right - observed as white (should be red for back)
                default_r, default_g, default_b = 1.0, 1.0, 1.0  # Force to red
                default_color_name = "white"
            elif i == 3:  # observed as white (should be red for back)
                default_r, default_g, default_b = 0.0, 0.0, 0.0  # Force to red
                default_color_name = "black"
            else:  # back_left - top LED (index 4)
                default_r, default_g, default_b = 1.0, 0.0, 0.0
                default_color_name = "red"

            if mode == "off":
                # Use default colors instead of turning off
                color.r = default_r
                color.g = default_g
                color.b = default_b
                color.a = 1.0
                color_name = default_color_name
            elif mode == "straight":
                # Use default colors for straight mode
                color.r = default_r
                color.g = default_g
                color.b = default_b
                color.a = 1.0
                color_name = default_color_name
            elif mode == "left":
                # Left turn signal: front_left LED (index 0) and back_left LED (index 2) blink yellow
                # Corrected mapping: assuming index 2 is back_left based on real vehicle behavior
                if (i == 0 or i == 4) and led_state:  # front_left and back_left LEDs
                    color.r = 1.0
                    color.g = 1.0
                    color.b = 0.0
                    color.a = 1.0
                    color_name = "yellow"
                else:
                    # Use default colors for non-blinking LEDs
                    color.r = default_r
                    color.g = default_g
                    color.b = default_b
                    color.a = 1.0
                    color_name = default_color_name
            elif mode == "right":
                # Right turn signal: front_right LED (index 1) and back_right LED (index 3) blink yellow
                # Corrected mapping: assuming index 3 is back_right based on real vehicle behavior
                if (i == 1 or i == 2) and led_state:  # front_right and back_right LEDs
                    color.r = 1.0
                    color.g = 1.0
                    color.b = 0.0
                    color.a = 1.0
                    color_name = "yellow"
                else:
                    # Use default colors for non-blinking LEDs
                    color.r = default_r
                    color.g = default_g
                    color.b = default_b
                    color.a = 1.0
                    color_name = default_color_name

            rgb_vals.append(color)
            color_list.append(color_name)

        msg.rgb_vals = rgb_vals
        msg.color_list = color_list

        # Color mask - apply to all LEDs
        msg.color_mask = [1, 1, 1, 1, 1]

        # Frequency (not used for this type of blinking)
        msg.frequency = 0.0

        # Frequency mask
        msg.frequency_mask = [0, 0, 0, 0, 0]

        return msg

    def blinker_command_callback(self, msg):
        """
        Callback for blinker command messages

        Args:
            msg (String): Command message with data "off", "left", or "right"
        """
        command = msg.data.lower().strip()

        if command in ["off", "left", "right", "straight"]:
            rospy.loginfo(f"{self._vehicle_name}: Received blinker command: {command}")

            # Stop current blinking
            if command == "off":
                rospy.sleep(2)
                # Stop current blinking immediately
                if self.blink_timer is not None:
                    self.blink_timer.shutdown()
                    self.blink_timer = None

                # Set LEDs to default colors (not off)
                led_msg = self.create_led_pattern("off", True)
                self.pub_led_pattern.publish(led_msg)
                rospy.loginfo(f"{self._vehicle_name}: LEDs set to default colors")
            elif command == "straight":
                # Stop current blinking immediately
                if self.blink_timer is not None:
                    self.blink_timer.shutdown()
                    self.blink_timer = None

                # Set LEDs to default colors for straight mode
                led_msg = self.create_led_pattern("straight", True)
                self.pub_led_pattern.publish(led_msg)
                rospy.loginfo(f"{self._vehicle_name}: LEDs set to straight mode (default colors)")
            else:
                # Stop current blinking immediately for new commands
                if self.blink_timer is not None:
                    self.blink_timer.shutdown()
                    self.blink_timer = None

                # Set current mode before starting timer
                self.current_mode = command

                # Start blinking for left or right
                self.is_on = True
                self.blink_timer = rospy.Timer(rospy.Duration(1.0 / (2.0 * self.blink_frequency)), self.blink_callback)
                rospy.loginfo(f"{self._vehicle_name}: Started {command} turn signal blinking")

            # Only set current_mode for off and straight modes here
            if command in ["off", "straight"]:
                self.current_mode = command

        else:
            rospy.logwarn(f"{self._vehicle_name}: Invalid blinker command: {command}. Use 'off', 'left', or 'right'")

    def blink_callback(self, event):
        """
        Timer callback for LED blinking
        """
        try:
            if self.current_mode in ["left", "right"]:
                # Create LED pattern for current mode and blink state
                led_msg = self.create_led_pattern(self.current_mode, self.is_on)
                self.pub_led_pattern.publish(led_msg)

                # Toggle state for next cycle
                self.is_on = not self.is_on

                state_text = "ON" if not self.is_on else "OFF"  # Show previous state
                rospy.logdebug(f"{self._vehicle_name}: {self.current_mode.upper()} blinker {state_text}")

        except Exception as e:
            rospy.logerr(f"{self._vehicle_name}: Error in blink callback: {e}")

    def start_blinking(self):
        """
        Start the LED blinking pattern (deprecated - now controlled by topic)
        """
        rospy.loginfo(f"{self._vehicle_name}: LED Blinker ready. Send commands to /{self._vehicle_name}/blinker_command")

    def stop_blinking(self):
        """
        Stop blinking and turn off all LEDs
        """
        rospy.loginfo(f"{self._vehicle_name}: Stopping LED blinking")

        # Turn off all LEDs
        led_msg = self.create_led_pattern("off", False)
        self.pub_led_pattern.publish(led_msg)

        # Stop timer
        if self.blink_timer is not None:
            self.blink_timer.shutdown()
            self.blink_timer = None

    def set_frequency(self, frequency):
        """
        Change the blink frequency

        Args:
            frequency (float): New frequency in Hz
        """
        self.blink_frequency = frequency

        # Restart timer with new frequency if currently blinking
        if self.blink_timer is not None and self.current_mode in ["left", "right"]:
            self.blink_timer.shutdown()
            self.blink_timer = rospy.Timer(rospy.Duration(1.0 / self.blink_frequency), self.blink_callback)

        rospy.loginfo(f"{self._vehicle_name}: Changed blink frequency to {frequency} Hz")

    def on_shutdown(self):
        """
        Cleanup when node shuts down
        """
        self.stop_blinking()
        rospy.loginfo(f"{self._vehicle_name}: LED Blinker Node shutting down")


if __name__ == "__main__":
    # Create the node
    node = LEDBlinkerNode(node_name="led_blinker_node")

    # Register shutdown hook
    rospy.on_shutdown(node.on_shutdown)

    # Initialize (ready to receive commands)
    node.start_blinking()

    try:
        # Keep the node running
        rospy.loginfo("LED Blinker Node ready. Send commands to topic.")
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("LED Blinker Node interrupted by user")
    except Exception as e:
        rospy.logerr(f"LED Blinker Node error: {e}")
    finally:
        node.stop_blinking()
