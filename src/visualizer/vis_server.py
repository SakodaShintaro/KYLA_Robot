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
        self.port_for_receiving_from_cam = 64851
        self.port_for_receiving_from_face_det = 64852
        self.buffer_size = 4096 * 4
        self.queue_size = 10
        self.image = None
        self.bbox_list = list()
        self.stop_threads = False

    def __update_cam_image(self):
        vis_socket_for_cam = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('Socket created (cam_server)')
        vis_socket_for_cam.bind((self.host, self.port_for_receiving_from_cam))
        print('Socket bind complete (cam_server)')
        vis_socket_for_cam.listen(self.queue_size)
        print('Socket now listening (cam_server)')

        conn_for_cam, addr = vis_socket_for_cam.accept()

        data = b""
        header_size = struct.calcsize(">L")
        while not self.stop_threads:
            while len(data) < header_size:
                data += conn_for_cam.recv(self.buffer_size)

            constant_sized_header = data[:header_size]
            payload_size = constant_sized_header
            data = data[header_size:]  # header 以降がペイロード（の一部。全部拾えてない事があるので）
            msg_size = struct.unpack(">L", payload_size)[0]
            # この時点ではヘッダーは拾えているが、ペイロードはまだ受け取れていない。

            # ペイロードを少しずつ受け取る。
            while len(data) < msg_size:
                recv_data = conn_for_cam.recv(self.buffer_size)
                data += recv_data
            frame_data = data[:msg_size]
            data = data[msg_size:]  # リセット

            image = np.fromstring(frame_data, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            self.image = image.copy()

        vis_socket_for_cam.close()


    def __update_face_bbox_list(self):
        vis_socket_for_face_det = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('Socket created (face_det_server)')
        vis_socket_for_face_det.bind((self.host, self.port_for_receiving_from_face_det))
        print('Socket bind complete (face_det_server)')
        vis_socket_for_face_det.listen(self.queue_size)
        print('Socket now listening (face_det_server)')
        conn_for_face_det, addr = vis_socket_for_face_det.accept()

        data = b""
        header_size = struct.calcsize(">L")
        while not self.stop_threads:
            while len(data) < header_size:
                data += conn_for_face_det.recv(self.buffer_size)

            constant_sized_header = data[:header_size]
            payload_size = constant_sized_header
            data = data[header_size:]  # header 以降がペイロード（の一部。全部拾えてない事があるので）
            msg_size = struct.unpack(">L", payload_size)[0]
            # この時点ではヘッダーは拾えているが、ペイロードはまだ受け取れていない。

            # ペイロードを少しずつ受け取る。
            while len(data) < msg_size:
                recv_data = conn_for_face_det.recv(self.buffer_size)
                data += recv_data
            frame_data = data[:msg_size]
            data = data[msg_size:]  # リセット
            self.bbox_list = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        # data += conn_for_face_det.recv(self.buffer_size)
        # self.image = None

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
                    cv2.rectangle(self.image, (sx, sy), (ex, ey), color, thickness)
                    cv2.putText(self.image, "face_id: " + str(arr_id), (sx, sy - 15), 1, 1.5, color, thickness)

                cv2.imshow('ImageWindow', self.image)
                key = cv2.waitKey(1)
                if key == 27:
                    cv2.destroyAllWindows()
                    # self.stop_threads = True
                    break


if __name__ == "__main__":
    vis_server = VisServer()
    vis_server.execute()
