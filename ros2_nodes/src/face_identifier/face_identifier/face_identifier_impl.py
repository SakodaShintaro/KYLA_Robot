import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
from glob import glob
from cv_bridge import CvBridge
import numpy as np
from face_region_msg.msg import FaceRegion
import torch
import torchvision
from .iresnet import iresnet100


class FaceIdentifierNode(Node):
    def __init__(self):
        super().__init__("face_identifier_node")
        # self.publisher = self.create_publisher(CompressedImage, "face_identifier_publisher", 10)
        self.timer = self.create_timer(0.1, self.on_tick)
        self.subscription = self.create_subscription(CompressedImage, "image_publisher", self.on_subscribe_image, 10)
        self.subscription = self.create_subscription(FaceRegion, "face_region", self.on_subscribe_region, 10)
        self.image_list_ = list()
        self.region_list_ = list()

        # IResNet config ---
        self.device = torch.device('cuda')
        self.model = iresnet100(pretrained=False)
        # ref: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch#model-zoo
        self.model.load_state_dict(torch.load(
            '/home/ubuntu/KYLA_Robot/src/models/backbone.pth',
            map_location=self.device))
        # --- IResNet config

    def on_tick(self):
        self.get_logger().info(f"Image : {len(self.image_list_)}, Regions : {len(self.region_list_)}")
        if len(self.image_list_) == 0 or len(self.region_list_) == 0:
            return

        curr_image = self.image_list_.pop()
        curr_region = self.region_list_.pop()

        curr_region = np.array(curr_region.region_points)
        curr_region = curr_region.reshape((-1, 4))

        image_height, image_width = curr_image.shape[0:2]
        image_list = list()
        for rect in curr_region:
            lux = int(rect[0] * image_width)
            luy = int(rect[1] * image_height)
            rdx = int(rect[2] * image_width)
            rdy = int(rect[3] * image_height)
            cv2.rectangle(curr_image, (lux, luy), (rdx, rdy), color=(0, 0, 255))
            image_list.append(curr_image[lux:rdx, luy:rdy])

        print(len(image_list))
        feature = self.infer(image_list)
        print(feature)

        # bridge = CvBridge()
        # cv2.imwrite("qwe.png", curr_image)
        # msg = bridge.cv2_to_compressed_imgmsg(curr_image)
        # self.publisher.publish(msg)

    def on_subscribe_image(self, msg):
        bridge = CvBridge()
        img = bridge.compressed_imgmsg_to_cv2(msg)
        self.image_list_.append(img)

    def on_subscribe_region(self, msg):
        self.region_list_.append(msg)

    def infer(self, img_list):
        image_np = np.array(img_list)
        print(image_np.shape)

        self.model.eval()
        self.model.to(self.device)
        image_tensor_list = list()

        image = self.__cv2pil(image_np)
        image = image.convert("RGB")
        image_tensor = torchvision.transforms.functional.to_tensor(image)
        image_tensor = torchvision.transforms.functional.resize(
            size=(112, 112), img=image_tensor)
        image_tensor = torch.unsqueeze(image_tensor, 0)
        image_tensor_list.append(image_tensor)

        torch_cat_image_tensor = torch.cat(image_tensor_list, 0)
        torch_cat_image_tensor = torch_cat_image_tensor.to(self.device)
        feat_list = self.model(torch_cat_image_tensor)

        processed_feat = [elem.item() for elem in feat_list[0]]
        ret_processed_feat = np.array(processed_feat)

        return ret_processed_feat


def main(args=None):
    rclpy.init(args=args)
    node = FaceIdentifierNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
