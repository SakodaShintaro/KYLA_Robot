import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
from cv_bridge import CvBridge
import sys
import numpy as np
from face_region_msg.msg import FaceRegion
from feature_extractor import FaceRoiExtractor


class MySubscriberNode(Node):
    def __init__(self):
        super().__init__("face_extractor_node")
        self.subscription = self.create_subscription(CompressedImage, "image_publisher", self.on_subscribe, 10)
        self.publisher = self.create_publisher(FaceRegion, "face_region", 10)
        self.face_extractor_ = FaceRoiExtractor(True)

    def on_subscribe(self, msg):
        bridge = CvBridge()
        img = bridge.compressed_imgmsg_to_cv2(msg)
        result = self.face_extractor_.execute(img)
        self.get_logger().info(f"Subscribe image")
        msg = FaceRegion()
        if result is None:
            msg.num = 0
            msg.region_points = list()
        else:
            msg.num = len(result) if result is not None else 0
            msg.region_points = list(np.array(result).flatten())
        self.publisher.publish(msg)
        self.get_logger().info(f"Publish msg")


def main(args=None):
    rclpy.init(args=args)
    node = MySubscriberNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
