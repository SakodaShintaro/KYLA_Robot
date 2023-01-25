import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
from glob import glob
from cv_bridge import CvBridge


class ImagePublisherNode(Node):
    def __init__(self):
        super().__init__("image_publisher_node")
        self.publisher = self.create_publisher(CompressedImage, "image_publisher", 10)
        self.timer = self.create_timer(0.1, self.on_tick)
        self.index = 0
        video = cv2.VideoCapture("/home/ubuntu/KYLA_Robot/assets/sample_movies/WIN_20221026_18_41_51_Pro.mp4")
        self.image_list = list()
        while True:
            result, frame = video.read()
            if not result:
                break
            self.image_list.append(frame)

    def on_tick(self):
        curr_image = self.image_list[self.index]
        bridge = CvBridge()
        msg = bridge.cv2_to_compressed_imgmsg(curr_image)
        self.get_logger().info(f"Publish image {self.index}")
        self.index += 1
        self.index %= len(self.image_list)
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
