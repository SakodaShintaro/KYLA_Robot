import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge


class MySubscriberNode(Node):
    def __init__(self):
        super().__init__("face_extractor_node")
        self.subscription = self.create_subscription(Image, "image_publisher", self.on_subscribe, 10)

    def on_subscribe(self, msg):
        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2.imwrite("qwe.png", img)
        self.get_logger().info(f"Subscribe image")


def main(args=None):
    rclpy.init(args=args)
    node = MySubscriberNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
