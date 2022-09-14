import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from glob import glob
from cv_bridge import CvBridge


class ImagePublisherNode(Node):
    def __init__(self):
        super().__init__("image_publisher_node")
        self.publisher = self.create_publisher(Image, "image_publisher", 10)
        self.timer = self.create_timer(1, self.on_tick)
        self.index = 0
        self.image_path_list = glob(f"/home/ubuntu/KYLA_Robot/src/feature_extractor/raw_data/*")
        self.image_path_list = sorted(self.image_path_list)

    def on_tick(self):
        curr_image_path = self.image_path_list[self.index]
        curr_image = cv2.imread(curr_image_path)
        bridge = CvBridge()
        msg = bridge.cv2_to_imgmsg(curr_image)
        self.index += 1
        self.index %= len(self.image_path_list)
        self.publisher.publish(msg)
        self.get_logger().info(f"Publish image")


def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
