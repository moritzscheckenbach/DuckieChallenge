import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


class CameraTriggerNode(Node):

    def __init__(self):
        super().__init__('camera_trigger_node')
        self.publisher_ = self.create_publisher(Bool, '/capture_trigger', 10)
        self.run()

    def get_key(self):
        key = input("Enter [p] to take a picture: ")
        return key

    def run(self):
        try:
            while rclpy.ok():
                key = self.get_key()
                if key == 'p':
                    msg = Bool()
                    msg.data = True
                    self.publisher_.publish(msg)
                    print("Bildaufnahme ausgelöst")
        except KeyboardInterrupt:
            print("Terminal wiederhergestellt.")
            self.get_logger().info("Node wurde beendet")
            self.destroy_node()


def main(args=None):
    rclpy.init(args=args)

    camera_trigger_node = CameraTriggerNode()
    rclpy.spin(camera_trigger_node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
