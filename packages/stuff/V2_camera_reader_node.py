#!/usr/bin/env python3

import os
import tkinter as tk

import cv2
import numpy as np
import rospy
import yaml
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from PIL import Image, ImageTk
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64


class CameraReaderNode(DTROS):

    def __init__(self, node_name):
        super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        # static parameters
        self._vehicle_name = os.environ["VEHICLE_NAME"]
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        # bridge between OpenCV and ROS
        self._bridge = CvBridge()

        with open("packages/followlane/config/detect_lane.yaml", "r") as f:
            text = f.read()
        self.conf = yaml.safe_load(text)

        self.lane_center = None

        self.create_window()

        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)

        self.sub_lane_center = rospy.Subscriber(f"/{self._vehicle_name}/detect/lane", Float64, self.lane_center_callback)
        self.sub_debug_image = rospy.Subscriber(f"/{self._vehicle_name}/detect/lane/debug/image/compressed", CompressedImage, self.debug_image_callback)

    def lane_center_callback(self, msg):
        # Store the latest lane center value
        self.lane_center = msg.data

    def callback(self, msg):

        if self.selected.get() != "lane_viz":
            self.update_conf()

        # convert JPEG bytes to CV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg)

        # create a copy for lane vizualization
        lane_viz_image = image.copy()

        # display frame
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # print(f'loaded image {image.shape}')
        if self.selected.get() == "lane_image":

            x_alt = 0
            y_alt = 0
            for point in ["top_left", "top_right", "bottom_left", "bottom_right", "top_left"]:
                x = self.conf["lane_image"][f"{point}_x"]
                y = self.conf["lane_image"][f"{point}_y"]

                if x_alt != 0 or y_alt != 0:
                    image = cv2.line(image, (x_alt, y_alt), (x, y), (255, 255, 255), 2)

                x_alt = x
                y_alt = y

        else:
            hl = self.conf[self.selected.get()]["hl"]
            hh = self.conf[self.selected.get()]["hh"]
            sl = self.conf[self.selected.get()]["sl"]
            sh = self.conf[self.selected.get()]["sh"]
            vl = self.conf[self.selected.get()]["vl"]
            vh = self.conf[self.selected.get()]["vh"]

            image = cv2.inRange(
                image,
                (hl, sl, vl),
                (hh, sh, vh),
            )

        # update image in window
        image = ImageTk.PhotoImage(Image.fromarray(image))
        self.panel.configure(image=image)
        self.panel.image = image

        # Create lane visualization if we have lane center data
        if self.lane_center is not None:
            # Create masked versions for white and yellow lanes
            white_mask = cv2.inRange(
                cv2.cvtColor(lane_viz_image, cv2.COLOR_BGR2HSV),
                (self.conf["white"]["hl"], self.conf["white"]["sl"], self.conf["white"]["vl"]),
                (self.conf["white"]["hh"], self.conf["white"]["sh"], self.conf["white"]["vh"]),
            )

            yellow_mask = cv2.inRange(
                cv2.cvtColor(lane_viz_image, cv2.COLOR_BGR2HSV),
                (self.conf["yellow"]["hl"], self.conf["yellow"]["sl"], self.conf["yellow"]["vl"]),
                (self.conf["yellow"]["hh"], self.conf["yellow"]["sh"], self.conf["yellow"]["vh"]),
            )

            # Apply masks to create colored lanes
            lane_viz_image[:, :, 0] = np.bitwise_or(lane_viz_image[:, :, 0], np.bitwise_and(yellow_mask, 255))
            lane_viz_image[:, :, 1] = np.bitwise_or(lane_viz_image[:, :, 1], np.bitwise_and(white_mask, 255))

            # Draw lane center line
            h, w = lane_viz_image.shape[:2]
            center_x = int(self.lane_center)
            cv2.line(lane_viz_image, (center_x, 0), (center_x, h), (0, 0, 255), 2)

            # Draw ideal center line (middle of image)
            middle_x = w // 2
            cv2.line(lane_viz_image, (middle_x, 0), (middle_x, h), (0, 255, 0), 1)

            # Update lane visualization if tab is selected
            if hasattr(self, "lane_viz_panel") and self.selected.get() == "lane_viz":
                lane_display = ImageTk.PhotoImage(Image.fromarray(lane_viz_image))
                self.lane_viz_panel.configure(image=lane_display)
                self.lane_viz_panel.image = lane_display

    def update_conf(self):
        # Skip configuration update for lane_viz since it doesn't have config values
        if self.selected.get() == "lane_viz":
            return

        for val in self.conf[self.selected.get()]:
            name = f"{self.selected.get()}_{val}"
            self.conf[self.selected.get()][val] = self.sliders[name].get()

    def debug_image_callback(self, msg):
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        if hasattr(self, "lane_viz_panel"):
            self.lane_viz_panel.configure(image=image)
            self.lane_viz_panel.image = image

    def print_conf(self):
        text = yaml.safe_dump(self.conf)
        print(f"#############\n{text}\n#############")

    # def change_menue(self, *args):
    #     self.slider_frame.pack_forget()
    #     print(f"selected menue : {self.selected.get()}")
    #     self.slider_frame = self.slider_frames[self.selected.get()]
    #     self.slider_frame.pack()
    #     self.print_conf()

    def change_menue(self, *args):
        self.slider_frame.pack_forget()
        print(f"selected menue : {self.selected.get()}")

        # Hide both panels first
        self.panel.pack_forget()
        self.lane_viz_panel.pack_forget()

        # Show appropriate panel based on selection
        if self.selected.get() == "lane_viz":
            self.lane_viz_panel.pack(side="bottom")
        else:
            self.panel.pack(side="bottom")

        # Show appropriate slider frame
        self.slider_frame = self.slider_frames[self.selected.get()]
        self.slider_frame.pack()

        self.print_conf()

    def create_window(self):
        self._root = tk.Tk()

        # Add Image For start only Black
        img = ImageTk.PhotoImage(Image.fromarray(np.zeros([480, 640, 3], np.uint8)))

        self.panel = tk.Label(self._root, image=img)
        self.panel.pack(side="bottom")

        # Create a separeate lane visualization panel
        self.lane_viz_panel = tk.Label(self._root, image=img)

        # Add drop down menu
        options = [s for s in self.conf]

        self.selected = tk.StringVar(self._root)
        self.selected.set(options[0])
        self.selected.trace("w", lambda *args: self.change_menue(*args))

        self.dropdown = tk.OptionMenu(self._root, self.selected, *options)
        self.dropdown.pack(side="top")

        # Add sliders
        self.sliders = {}
        self.slider_frames = {}
        for option in self.conf:
            frame = tk.Frame(self._root)
            for val in self.conf[option]:
                name = f"{option}_{val}"

                if option == "lane_image":
                    self.sliders[name] = tk.Scale(frame, from_=-100, to=700, orient="horizontal", label=val)
                else:
                    self.sliders[name] = tk.Scale(frame, from_=0, to=255, orient="horizontal", label=val)
                self.sliders[name].set(self.conf[option][val])
                self.sliders[name].pack(side="left")

            self.slider_frames[option] = frame

        self.slider_frames["lane_viz"] = tk.Frame(self._root)

        self.slider_frame = self.slider_frames[self.selected.get()]
        self.slider_frame.pack()

    def run(self):
        self._root.mainloop()
        self.print_conf()
        rospy.signal_shutdown("User endet Programm")


if __name__ == "__main__":
    # create the node
    node = CameraReaderNode(node_name="camera_reader_node")
    # keep spinning
    node.run()
    rospy.spin()
