import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
import cv2
from cv_bridge import CvBridge
import numpy as np
from face_region_msg.msg import FaceRegion
import torch
import torchvision
from .iresnet import iresnet100
from PIL import Image as PImage
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

        self.list_of_feature_ = list()
        self.next_id_ = 0

        # IResNet config ---
        self.device = torch.device('cuda')
        self.model = iresnet100(pretrained=False)
        # ref: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch#model-zoo
        self.model.load_state_dict(torch.load(
            '/home/ubuntu/KYLA_Robot/assets/models/backbone.pth',
            map_location=self.device))
        # --- IResNet config

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
            self.list_of_feature_.clear()
            return

        result = self.face_matcher_(image_list)
        for sim, name in result:
            self.get_logger().info(f"sim = {sim:.4f}, name = {name}")

        feature = self.infer(image_list)
        self.get_logger().info(f"len(feature) = {len(feature)}")

        next_feature_list = list()

        for i, feat in enumerate(feature):
            max_value = -float("inf")
            max_index = -1

            curr_id = None

            if max_index == -1:
                curr_id = self.next_id_
                self.next_id_ += 1
            else:
                curr_id = self.list_of_feature_[max_index]

            next_feature_list.append((feat, curr_id))
            lux, luy, rdx, rdy = region_list[i]
            sim = result[i][0]
            name = result[i][1] if sim > 0.3 else "unknown" 
            cv2.putText(curr_image, f"{name}({sim:.3f})",
                        org=(rdx, rdy),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.0,
                        color=(0, 255, 0),
                        thickness=2,
                        lineType=cv2.LINE_4)

        self.list_of_feature_ = next_feature_list

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

    def infer(self, img_list):
        self.model.eval()
        self.model.to(self.device)
        image_tensor_list = list()

        for img in img_list:
            image = self.__cv2pil(img)
            image = image.convert("RGB")
            image_tensor = torchvision.transforms.functional.to_tensor(image)
            image_tensor = torchvision.transforms.functional.resize(
                size=(112, 112), img=image_tensor)
            image_tensor = torch.unsqueeze(image_tensor, 0)
            image_tensor_list.append(image_tensor)

        torch_cat_image_tensor = torch.cat(image_tensor_list, 0)
        torch_cat_image_tensor = torch_cat_image_tensor.to(self.device)
        feat_list = self.model(torch_cat_image_tensor)

        result = list()
        for feat in feat_list:
            processed_feat = [elem.item() for elem in feat]
            ret_processed_feat = np.array(processed_feat)
            result.append(ret_processed_feat)
        return result

    def __cv2pil(self, image):
        ''' OpenCV型 -> PIL型 '''
        new_image = image.copy()
        if new_image.ndim == 2:  # モノクロ
            pass
        elif new_image.shape[2] == 3:  # カラー
            new_image = new_image[:, :, ::-1]
        elif new_image.shape[2] == 4:  # 透過
            new_image = new_image[:, :, [2, 1, 0, 3]]
        new_image = PImage.fromarray(new_image)
        return new_image


def main(args=None):
    rclpy.init(args=args)
    node = FaceIdentifierNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
