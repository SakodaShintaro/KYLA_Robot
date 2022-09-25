# -*- coding: utf-8 -*-
import cv2
import json
import socket
import struct
from base_server import BaseServer

class CamServer(BaseServer):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.frame_id = 0

        # # setting server ---
        # self.face_det_port_id = 50501
        # self.face_det_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.face_det_socket.bind(("localhost", self.face_det_port_id))  # IPとポート番号を指定します
        # self.face_det_socket.listen(5)
        # self.face_det_buffer_size = 1024
        # # ---

        # to face_det_server ---
        self.face_det_port_id = 50502
        self.face_det_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.face_det_socket.settimeout(0.1)    # タイムアウト値設定を追加する
        self.face_det_socket.connect(("localhost", self.face_det_port_id))
        self.face_det_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # self.face_det_buffer_size = 1024
        # ---

    def update(self):
        ret, image = self.cap.read()
        if not ret:
            return

        self.frame_id += 1

        # send image-info to face_det_server ---
        fmt = "d"
        send_data1 = struct.pack(fmt, self.frame_id)
        self.face_det_socket.send(send_data1)
        send_data2 = struct.pack(fmt, image.shape[1])  # width
        self.face_det_socket.send(send_data2)
        send_data3 = struct.pack(fmt, image.shape[0])  # height
        self.face_det_socket.send(send_data3)
        send_data4 = image.tostring()
        self.face_det_socket.send(send_data4)
        # ---

        self.face_det_socket.close()


if __name__ == "__main__":
    cam_server = CamServer()
    cam_server.execute()