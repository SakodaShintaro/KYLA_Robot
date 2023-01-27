import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import numpy as np


class ThermographyPublisherNode(Node):
    def __init__(self):
        super().__init__("thermography_publisher_node")
        self.publisher = self.create_publisher(CompressedImage, "thermography_publisher", 10)
        self.timer = self.create_timer(0.1, self.on_tick)
        self.H = 8
        self.W = 8

    def on_tick(self):
        curr_image = np.ones((self.H, self.W)) * 36
        bridge = CvBridge()
        msg = bridge.cv2_to_compressed_imgmsg(curr_image)
        self.get_logger().info(f"Publish thermography")
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ThermographyPublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
