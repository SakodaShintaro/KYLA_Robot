# -*- coding: utf-8 -*-
import socket
import json
import struct
import cv2
import numpy as np
from base_server import BaseServer

class FaceDetServer(BaseServer):
    def __init__(self):
        super().__init__()

        # server ---
        self.port_id = 50502
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(("localhost", self.port_id))  # IPとポート番号を指定します
        self.socket.listen(5)
        self.buffer_size = 1024
        # ---

        # from cam_server ---
        # self.cam_buffer_size = 1024
        self.cam_buffer_size = 2048000
        # ---

        # # server ---
        # self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.socket.connect((socket.gethostname(), 50501))
        # self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # self.buffer_size = 1024
        # # ---


    def update(self):
        client_socket, address = self.socket.accept()
        fmt = "d"
        unpacker = struct.Struct(fmt)
        recv_data1 = client_socket.recv(unpacker.size, socket.MSG_WAITALL)
        frame_id = unpacker.unpack(recv_data1)[0]
        recv_data2 = client_socket.recv(unpacker.size, socket.MSG_WAITALL)
        width = unpacker.unpack(recv_data2)[0]
        recv_data3 = client_socket.recv(unpacker.size, socket.MSG_WAITALL)
        height = unpacker.unpack(recv_data3)[0]

        width = int(width)
        height = int(height)

        # frame_data = frame_data.decode("utf-8").rstrip("\x00")
        # frame_data = json.loads(frame_data)

        image = client_socket.recv(
            width * height * 3, socket.MSG_WAITALL
        )
        image = np.fromstring(image, dtype=np.uint8)
        image = np.reshape(image, (height, width, 3))

        # frame_data = frame_data.fromstring()
        cv2.imshow("image", image)
        cv2.waitKey(1000)

        # print(frame_data, self.frame_data.qsize())
        # self.frame_data.put(frame_data)


if __name__ == "__main__":
    cam_server = FaceDetServer()
    cam_server.execute()