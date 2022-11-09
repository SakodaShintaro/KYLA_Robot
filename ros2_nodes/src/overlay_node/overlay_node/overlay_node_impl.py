import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
from glob import glob
from cv_bridge import CvBridge
import numpy as np
from face_region_msg.msg import FaceRegion


class OverlayNode(Node):
    def __init__(self):
        super().__init__("overlay_node")
        self.publisher = self.create_publisher(CompressedImage, "overlay_publisher", 10)
        self.timer = self.create_timer(0.1, self.on_tick)
        self.subscription = self.create_subscription(CompressedImage, "image_publisher", self.on_subscribe_image, 10)
        self.subscription = self.create_subscription(FaceRegion, "face_region", self.on_subscribe_region, 10)
        self.image_list_ = list()
        self.region_list_ = list()

    def on_tick(self):
        self.get_logger().info(f"Image : {len(self.image_list_)}, Regions : {len(self.region_list_)}")
        if len(self.image_list_) == 0 or len(self.region_list_) == 0:
            return

        curr_image = self.image_list_.pop()
        curr_region = self.region_list_.pop()

        curr_region = np.array(curr_region.region_points)
        curr_region = curr_region.reshape((-1, 4))

        image_height, image_width = curr_image.shape[0:2]
        for rect in curr_region:
            lux = int(rect[0] * image_width)
            luy = int(rect[1] * image_height)
            rdx = int(rect[2] * image_width)
            rdy = int(rect[3] * image_height)
            cv2.rectangle(curr_image, (lux, luy), (rdx, rdy), color=(0, 0, 255))

        bridge = CvBridge()
        cv2.imwrite("qwe.png", curr_image)
        msg = bridge.cv2_to_compressed_imgmsg(curr_image)
        self.publisher.publish(msg)

    def on_subscribe_image(self, msg):
        bridge = CvBridge()
        img = bridge.compressed_imgmsg_to_cv2(msg)
        self.image_list_.append(img)

    def on_subscribe_region(self, msg):
        self.region_list_.append(msg)


def main(args=None):
    rclpy.init(args=args)
    node = OverlayNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
