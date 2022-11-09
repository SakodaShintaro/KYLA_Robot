# -*- coding: utf-8 -*-
import cv2
import socket
import struct
import threading
import numpy as np
import pickle

import torch
import torchvision
from PIL import Image
from iresnet import iresnet100
import numpy as np

from PIL import Image


class FaceReIDServer(object):
    def __init__(self):
        self.host = "localhost"
        self.buffer_size = 4096
        self.queue_size = 10

        self.port_for_receiving_image_from_cam = 64854
        self.vis_socket_for_receiving_cam_image = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        self.vis_socket_for_receiving_cam_image.bind(
            (self.host, self.port_for_receiving_image_from_cam))
        self.vis_socket_for_receiving_cam_image.listen(self.queue_size)

        self.port_for_receiving_bbox_from_face_det = 64855
        self.vis_socket_for_receiving_bbox_face_det = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        self.vis_socket_for_receiving_bbox_face_det.bind(
            (self.host, self.port_for_receiving_bbox_from_face_det))
        self.vis_socket_for_receiving_bbox_face_det.listen(self.queue_size)

        # self.port_for_sending_to_vis = 64856
        # self.client_socket_for_vis = socket.socket(
        #     socket.AF_INET, socket.SOCK_STREAM)
        # self.client_socket_for_vis.connect(
        #     (self.host, self.port_for_sending_to_vis))

        self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

        self.bbox_list = list()
        self.fresh_image = None

        # IResNet config ---
        self.device = torch.device('cuda')
        self.model = iresnet100(pretrained=False)

        # ref: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch#model-zoo
        self.model.load_state_dict(torch.load(
            '../models/backbone.pth',
            map_location=self.device))
        # --- IResNet config

    def __cv2pil(self, image):
        ''' OpenCV型 -> PIL型 
        Comments:
            共通化したい。
        '''
        new_image = image.copy()
        if new_image.ndim == 2:  # モノクロ
            pass
        elif new_image.shape[2] == 3:  # カラー
            new_image = new_image[:, :, ::-1]
        elif new_image.shape[2] == 4:  # 透過
            new_image = new_image[:, :, [2, 1, 0, 3]]
        new_image = Image.fromarray(new_image)
        return new_image

    def __extract_reid_feature(self, image_np):

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

        # ret_processed_feat_list = list()
        # for i in range(len(feat_list)):
        processed_feat = [elem.item() for elem in feat_list[0]]
        ret_processed_feat = np.array(processed_feat)

        # print(feat_list[0].shape)
        return ret_processed_feat

    def __update_cam_image(self):
        conn_for_cam, addr = self.vis_socket_for_receiving_cam_image.accept()
        # print(conn_for_cam)

        data = b""
        header_size = struct.calcsize(">L")
        while True:
            # ヘッダーをまず読み取って、画像データのサイズを調べる
            while len(data) < header_size:
                data += conn_for_cam.recv(self.buffer_size)
            # print(len(data))

            constant_sized_header = data[:header_size]
            payload_size_bytes = constant_sized_header
            data = data[header_size:]  # header 以降がペイロード（の一部。全部拾えてない事があるので）
            payload_size = struct.unpack(">L", payload_size_bytes)[0]

            # 画像サイズ分だけペイロードを少しずつ受け取る。
            # ref: https://gist.github.com/kittinan/e7ecefddda5616eab2765fdb2affed1b
            while len(data) < payload_size:
                recv_data = conn_for_cam.recv(self.buffer_size)
                data += recv_data
            image_bytes = data[:payload_size]
            data = data[payload_size:]  # 先頭位置をずらす

            image = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            self.fresh_image = image.copy()

        # 止めるときはキルするので下記は実行されない。
        self.vis_socket_for_receiving_cam_image.close()

    def __update_face_bbox_list(self):
        conn_for_face_det, addr = self.vis_socket_for_receiving_bbox_face_det.accept()

        data = b""
        header_size = struct.calcsize(">L")
        while True:
            # ヘッダーをまず読み取って、画像データのサイズを調べる
            while len(data) < header_size:
                data += conn_for_face_det.recv(self.buffer_size)

            constant_sized_header = data[:header_size]
            payload_size_bytes = constant_sized_header
            data = data[header_size:]  # header 以降がペイロード（の一部。全部拾えてない事があるので）
            payload_size = struct.unpack(">L", payload_size_bytes)[0]

            # 画像サイズ分だけペイロードを少しずつ受け取る。
            # ref: https://gist.github.com/kittinan/e7ecefddda5616eab2765fdb2affed1b
            while len(data) < payload_size:
                recv_data = conn_for_face_det.recv(self.buffer_size)
                data += recv_data
            bbox_list_bytes = data[:payload_size]
            data = data[payload_size:]  # 先頭位置をずらす
            self.bbox_list = pickle.loads(
                bbox_list_bytes, fix_imports=True, encoding="bytes")

        # 止めるときはキルするので下記は実行されない。
        self.vis_socket_for_face_det.close()

    def __update_croppepd_img_list(self):
        # face_det_cnt = 0
        while True:
            # 別スレッドでの画像取得が終わるまでは None なので飛ばす。
            if self.fresh_image is None:
                # print("fresh_image is None.")
                continue

            if self.bbox_list is list():
                # print("bbox_list is None.")
                continue

            # print("Progress.")

            cropped_image_list = list()
            for bbox in self.bbox_list:
                sx = bbox[0]
                sy = bbox[1]
                ex = bbox[2]
                ey = bbox[3]
                croppted_fresh_image = self.fresh_image[sy:ey, sx:ex]
                # cv2.imwrite("croppted_fresh_image.png", croppted_fresh_image)
                cropped_image_list.append(croppted_fresh_image)

                reid_feature = self.__extract_reid_feature(croppted_fresh_image)
                print(reid_feature)

            size_of_cropped_image_list = len(cropped_image_list)
            # print("{}: {}".format(size_of_frame))

            # # 決まったサイズでヘッダーをつけて、受け取り側でペイロードの大きさが分かるようにする。
            # # ref: https://gist.github.com/kittinan/e7ecefddda5616eab2765fdb2affed1b
            # constant_sized_header = struct.pack(">L", size_of_cropped_image_list)  # ビックエンディアンで 4byte のサイズ変数を作る
            # cropped_image_list_bytes = pickle.dumps(cropped_image_list, 0)

            # # ヘッダーをつけてバイト列を送信
            # # self.client_socket_for_vis2.sendall(
            # #     constant_sized_header + cropped_image_list_bytes)

            # # 新鮮な画像が取れたら顔検出する。

        # 止めるときはキルするので下記は実行されない。
        self.client_socket_for_vis.close()


    def execute(self):
        th_for_cam = threading.Thread(target=self.__update_cam_image)
        th_for_cam.start()
        th_for_bbox = threading.Thread(target=self.__update_face_bbox_list)
        th_for_bbox.start()
        th_for_cropped_img = threading.Thread(target=self.__update_croppepd_img_list)
        th_for_cropped_img.start()


if __name__ == "__main__":
    cam_server = FaceReIDServer()
    cam_server.execute()
