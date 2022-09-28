# -*- coding: utf-8 -*-
import cv2
import socket
import struct


class CamServer(object):
    def __init__(self):
        self.host = "localhost"
        self.port_for_sending_to_face_det = 64850
        self.client_socket_for_face_det = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket_for_face_det.connect(
            (self.host, self.port_for_sending_to_face_det))

        self.port_for_sending_to_vis = 64851
        self.client_socket_for_vis = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket_for_vis.connect(
            (self.host, self.port_for_sending_to_vis))

        self.cap = cv2.VideoCapture(0)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        # print(self.fps)
        self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

    def execute(self):
        while self.cap.isOpened():
            ret, image = self.cap.read()
            if not ret:
                continue
            ret, image = cv2.imencode(".jpg", image, self.encode_param)  # np.array (dim1) へ変換
            size_of_image = image.shape[0]
            # print("{}: {}".format(size_of_frame))

            # 決まったサイズでヘッダーをつけて、受け取り側でペイロードの大きさが分かるようにする。
            # ref: https://gist.github.com/kittinan/e7ecefddda5616eab2765fdb2affed1b
            constant_sized_header = struct.pack(">L", size_of_image)  # ビックエンディアンで 4byte のサイズ変数を作る
            frame_data = image.tobytes()  # ndarray から純粋なバイト列に変換

            # ヘッダーをつけてバイト列を送信
            self.client_socket_for_face_det.sendall(
                constant_sized_header + frame_data)
            self.client_socket_for_vis.sendall(
                constant_sized_header + frame_data)
            # time.sleep(1.0 / fps)

        # 止めるときはキルするので下記は実行されない。
        self.client_socket_for_face_det.close()
        self.client_socket_for_vis.close()
        self.cap.release()


if __name__ == "__main__":
    cam_server = CamServer()
    cam_server.execute()
