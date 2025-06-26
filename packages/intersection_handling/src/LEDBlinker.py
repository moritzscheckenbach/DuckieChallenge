#!/usr/bin/env python3

import os

import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import LEDPattern
from std_msgs.msg import ColorRGBA, Header


class LEDBlinkerNode(DTROS):
    def __init__(self, node_name):
        super(LEDBlinkerNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle_name = os.environ.get("VEHICLE_NAME", "dorette")

        # LED pattern publisher
        self.pub_led_pattern = rospy.Publisher(f"/{self._vehicle_name}/led_emitter_node/led_pattern", LEDPattern, queue_size=1)

        # Blink parameters
        self.blink_frequency = 2.0  # 2 Hz = 0.5 seconds on/off
        self.is_on = True

        # Timer for blinking
        self.blink_timer = rospy.Timer(rospy.Duration(1.0 / self.blink_frequency), self.blink_callback)

        rospy.loginfo(f"{self._vehicle_name}: LED Blinker Node initialized")

    def create_led_pattern(self, yellow_on=True):
        """
        Create LED pattern message for yellow blinking

        Args:
            yellow_on (bool): True for yellow, False for off (black)
        """
        msg = LEDPattern()

        # Header
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = ""

        # Color definitions - using color names
        if yellow_on:
            msg.color_list = ["yellow", "yellow", "yellow", "yellow", "yellow"]
        else:
            msg.color_list = ["off", "off", "off", "off", "off"]

        # RGB values for yellow (R=1, G=1, B=0) or off (R=0, G=0, B=0)
        rgb_vals = []
        for i in range(5):  # 5 LEDs
            color = ColorRGBA()
            if yellow_on:
                color.r = 1.0
                color.g = 1.0
                color.b = 0.0
                color.a = 1.0
            else:
                color.r = 0.0
                color.g = 0.0
                color.b = 0.0
                color.a = 1.0
            rgb_vals.append(color)

        msg.rgb_vals = rgb_vals

        # Color mask - apply to all LEDs (1 = active, 0 = inactive)
        msg.color_mask = [1, 1, 1, 1, 1]  # All 5 LEDs active

        # Frequency (not used for this type of blinking)
        msg.frequency = 0.0

        # Frequency mask
        msg.frequency_mask = [0, 0, 0, 0, 0]

        return msg

    def blink_callback(self, event):
        """
        Timer callback for LED blinking
        """
        try:
            # Toggle LED state
            led_msg = self.create_led_pattern(self.is_on)
            self.pub_led_pattern.publish(led_msg)

            # Toggle state for next cycle
            self.is_on = not self.is_on

            state_text = "ON" if not self.is_on else "OFF"  # Show previous state
            rospy.loginfo(f"{self._vehicle_name}: LEDs {state_text}")

        except Exception as e:
            rospy.logerr(f"{self._vehicle_name}: Error in blink callback: {e}")

    def start_blinking(self):
        """
        Start the LED blinking pattern
        """
        rospy.loginfo(f"{self._vehicle_name}: Starting LED blinking at {self.blink_frequency} Hz")

    def stop_blinking(self):
        """
        Stop blinking and turn off all LEDs
        """
        rospy.loginfo(f"{self._vehicle_name}: Stopping LED blinking")

        # Turn off all LEDs
        led_msg = self.create_led_pattern(False)
        self.pub_led_pattern.publish(led_msg)

        # Stop timer
        if hasattr(self, "blink_timer"):
            self.blink_timer.shutdown()

    def set_frequency(self, frequency):
        """
        Change the blink frequency

        Args:
            frequency (float): New frequency in Hz
        """
        self.blink_frequency = frequency

        # Restart timer with new frequency
        if hasattr(self, "blink_timer"):
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

    # Start blinking
    node.start_blinking()

    try:
        # Keep the node running
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("LED Blinker Node interrupted by user")
    except Exception as e:
        rospy.logerr(f"LED Blinker Node error: {e}")
    finally:
        node.stop_blinking()
