# -*- coding: utf-8 -*-
# ref: https://gist.github.com/kittinan/e7ecefddda5616eab2765fdb2affed1b
from msilib.schema import Feature
import socket
import sys
import cv2
import pickle
import numpy as np
import struct  # new
import threading
# import glob
import mediapipe as mp


class VisServer(object):
    def __init__(self):
        self.host = 'localhost'
        self.buffer_size = 4096
        self.queue_size = 10
        self.image = None
        self.bbox_list = list()

        self.port_for_receiving_from_cam = 64851
        self.vis_socket_for_cam = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        self.vis_socket_for_cam.bind(
            (self.host, self.port_for_receiving_from_cam))
        self.vis_socket_for_cam.listen(self.queue_size)

        self.port_for_receiving_from_face_det = 64852
        self.vis_socket_for_face_det = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        self.vis_socket_for_face_det.bind(
            (self.host, self.port_for_receiving_from_face_det))
        self.vis_socket_for_face_det.listen(self.queue_size)

        print('Setting Sockets')

    def __update_cam_image(self):
        conn_for_cam, addr = self.vis_socket_for_cam.accept()

        data = b""
        header_size = struct.calcsize(">L")
        while True:
            # ヘッダーをまず読み取って、画像データのサイズを調べる
            while len(data) < header_size:
                data += conn_for_cam.recv(self.buffer_size)

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
            self.image = image.copy()

        # 止めるときはキルするので下記は実行されない。
        self.vis_socket_for_cam.close()

    def __update_face_bbox_list(self):
        conn_for_face_det, addr = self.vis_socket_for_face_det.accept()

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

    def execute(self):
        # queue = queue.Queue(maxsize=1)
        th_for_cam = threading.Thread(target=self.__update_cam_image)
        th_for_cam.start()
        th_for_face_det = threading.Thread(target=self.__update_face_bbox_list)
        th_for_face_det.start()

        while True:
            if self.image is not None:
                for arr_id, bbox in enumerate(self.bbox_list):
                    sx, sy, ex, ey = bbox
                    color = (255, 255, 0)
                    thickness = 2
                    cv2.rectangle(self.image, (sx, sy),
                                  (ex, ey), color, thickness)
                    cv2.putText(self.image, "face_id: " + str(arr_id),
                                (sx, sy - 15), 1, 1.5, color, thickness)

                cv2.imshow('ImageWindow', self.image)
                key = cv2.waitKey(1)
                if key == 27:
                    cv2.destroyAllWindows()
                    break


if __name__ == "__main__":
    vis_server = VisServer()
    vis_server.execute()