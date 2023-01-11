import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
import cv2
from cv_bridge import CvBridge
import numpy as np
from face_region_msg.msg import FaceRegion
from feature_extractor import FaceMatcher


class FaceIdentifierNode(Node):
    def __init__(self):
        super().__init__("face_identifier_node")
        self.publisher = self.create_publisher(Image, "face_identifier_publisher", 10)
        self.timer = self.create_timer(0.1, self.on_tick)
        self.subscription = self.create_subscription(CompressedImage, "image_publisher", self.on_subscribe_image, 10)
        self.subscription = self.create_subscription(FaceRegion, "face_region", self.on_subscribe_region, 10)
        self.image_list_ = list()
        self.region_list_ = list()
        self.face_matcher_ = FaceMatcher('/home/ubuntu/KYLA_Robot/assets/database/FACE_FEATURES.db')

    def on_tick(self):
        self.get_logger().info(f"Image : {len(self.image_list_)}, Regions : {len(self.region_list_)}")
        if len(self.image_list_) == 0 or len(self.region_list_) == 0:
            return

        # TODO: 計算量的に悪くないか確認する
        curr_image = self.image_list_.pop(0)
        curr_region = self.region_list_.pop(0)

        curr_region = np.array(curr_region.region_points)
        curr_region = curr_region.reshape((-1, 4))

        self.get_logger().info(f"curr_image.shape = {curr_image.shape}")

        region_list = list()

        image_height, image_width = curr_image.shape[0:2]
        image_list = list()
        for rect in curr_region:
            lux = int(rect[0] * image_width)
            luy = int(rect[1] * image_height)
            rdx = int(rect[2] * image_width)
            rdy = int(rect[3] * image_height)
            cv2.rectangle(curr_image, (lux, luy), (rdx, rdy), color=(0, 0, 255))
            push_image = curr_image[luy:rdy, lux:rdx]
            self.get_logger().info(f"lux = {lux}, luy = {luy}, rdx = {rdx}, rdy = {rdy}")
            self.get_logger().info(f"push_image.shape = {push_image.shape}")
            image_list.append(push_image)
            region_list.append((lux, luy, rdx, rdy))

        self.get_logger().info(f"len(image_list) = {len(image_list)}")

        if len(image_list) == 0:
            bridge = CvBridge()
            curr_image = cv2.rotate(curr_image, cv2.ROTATE_90_CLOCKWISE)
            msg = bridge.cv2_to_imgmsg(curr_image, encoding="bgr8")
            self.publisher.publish(msg)
            return

        result = self.face_matcher_(image_list)
        for i in range(len(region_list)):
            lux, luy, rdx, rdy = region_list[i]
            sim = result[i][0]
            unknown = sim <= 0.3
            name = "unknown" if unknown else result[i][1]
            color = (0, 0, 255) if unknown else (0, 255, 0)

            self.get_logger().info(f"sim = {sim:.4f}, name = {name}")
            cv2.putText(curr_image, f"{name}({sim:.3f})",
                        org=(rdx, rdy),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.0,
                        color=color,
                        thickness=2,
                        lineType=cv2.LINE_4)

        bridge = CvBridge()
        curr_image = cv2.rotate(curr_image, cv2.ROTATE_90_CLOCKWISE)
        msg = bridge.cv2_to_imgmsg(curr_image, encoding="bgr8")
        self.publisher.publish(msg)

    def on_subscribe_image(self, msg):
        bridge = CvBridge()
        img = bridge.compressed_imgmsg_to_cv2(msg)
        self.image_list_.append(img)

    def on_subscribe_region(self, msg):
        self.region_list_.append(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FaceIdentifierNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
