# -*- coding: utf-8 -*- 
# ref: https://gist.github.com/kittinan/e7ecefddda5616eab2765fdb2affed1b
import cv2
import io
import socket
import struct
import time
import pickle
import zlib
import time
import numpy as np

class CamServer(object):
    def __init__(self):
        self.port_for_sending_to_face_det = 64850
        self.client_socket_for_face_det = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket_for_face_det.connect(('localhost', self.port_for_sending_to_face_det))
        # self.connection = self.client_socket_for_face_det.makefile('wb')

        self.port_for_sending_to_vis = 64851
        self.client_socket_for_vis = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket_for_vis.connect(('localhost', self.port_for_sending_to_vis))

        # video_file = "./face_mesh_input.mp4"
        # cam = cv2.VideoCapture(video_file)
        self.cam = cv2.VideoCapture(0)

        self.fps = self.cam.get(cv2.CAP_PROP_FPS)
        # print(self.fps)
        self.img_counter = 0
        self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

    def execute(self):
        while self.cam.isOpened():
            ret, frame = self.cam.read()
            if not ret:
                continue
            # frame = np.array(frame, dtype=np.uint8)
            ret, frame = cv2.imencode('.jpg', frame, self.encode_param)
            #    data = zlib.compress(pickle.dumps(frame, 0))
            size_of_frame = frame.shape[0]
            print("{}: {}".format(self.img_counter, size_of_frame))
            # client_socket.sendall(struct.pack(">L", size_of_data) + data)

            # 決まったサイズでヘッダーをつけて、受け取り側でペイロードの大きさが分かるようにする。
            constant_sized_header = struct.pack(">L", size_of_frame)
            frame_data = frame.tobytes()  # ndarray から純粋なバイト列に変換
            self.client_socket_for_face_det.sendall(constant_sized_header + frame_data)
            self.client_socket_for_vis.sendall(constant_sized_header + frame_data)
            # client_socket.sendall(frame)
            self.img_counter += 1
            # time.sleep(1.0 / fps)

        self.client_socket_for_face_det.close()
        self.cam.release()

if __name__ == "__main__":
    cam_server = CamServer()
    cam_server.execute()

