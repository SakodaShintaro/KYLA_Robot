# -*- coding: utf-8 -*-
import socket
import sys
import cv2
import pickle
import numpy as np
import struct  # new
import threading
# import glob
import mediapipe as mp


class FaceDetServer(object):
    def __init__(self, video_file=None):
        self.host = "localhost"
        self.buffer_size = 4096
        self.queue_size = 10
        self.fresh_image = None
        self.bbox_list = list()

        self.port_for_receiving_from_cam = 64850
        self.vis_socket_for_cam = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        self.vis_socket_for_cam.bind(
            (self.host, self.port_for_receiving_from_cam))
        self.vis_socket_for_cam.listen(self.queue_size)

        self.port_for_sending_to_vis = 64852
        self.client_socket_for_vis = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket_for_vis.connect(
            (self.host, self.port_for_sending_to_vis))

        self.port_for_sending_bbox_list_to_reid = 64855
        self.client_socket_for_reid = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket_for_reid.connect(
            (self.host, self.port_for_sending_bbox_list_to_reid))

        # for camera
        self.port_for_sending_to_vis2 = 64851
        self.client_socket_for_vis2 = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket_for_vis2.connect(
            (self.host, self.port_for_sending_to_vis2))

        self.port_for_sending_to_reid2 = 64854
        self.client_socket_for_reid2 = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket_for_reid2.connect(
            (self.host, self.port_for_sending_to_reid2))
        self.video_file = video_file
        if self.video_file is None:
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(self.video_file)
        self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

        print('Setting Sockets')

    def __update_face_bbox_list(self):
        # face_det_cnt = 0
        # while True:
        frame_id = 0
        while self.cap.isOpened():
            ret, self.fresh_image = self.cap.read()
            # print(self.fresh_image.shape)
            # self.fresh_image = self.fresh_image[self.fresh_image.shape[0] // 2: :, :]
            # self.fresh_image = cv2.rotate(self.fresh_image, cv2.ROTATE_180)
            # image = cv2.rotate(image, cv2.ROTATE_180)
            if not ret:
                if self.video_file is not None:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset
                continue

            # 決まったサイズでヘッダーをつけて、受け取り側でペイロードの大きさが分かるようにする。
            # ref: https://gist.github.com/kittinan/e7ecefddda5616eab2765fdb2affed1b
            # print("Progress.")

            # 新鮮な画像が取れたら顔検出する。
            enable_expand_roi = True
            try:
                mp_face_detection = mp.solutions.face_detection
                mp_drawing = mp.solutions.drawing_utils

                with mp_face_detection.FaceDetection(
                        model_selection=1, min_detection_confidence=0.2) as face_detection:
                    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
                    results = face_detection.process(
                        cv2.cvtColor(self.fresh_image, cv2.COLOR_BGR2RGB))

                    # Draw face detections of each face.
                    if results.detections:
                        self.bbox_list = list()
                        self.bbox_list.append(frame_id)
                        # roi_image = image.copy()
                        image_rows, image_cols, _ = self.fresh_image.shape
                        for detection in results.detections:
                            location = detection.location_data
                            # keypoint_list = location.relative_keypoints

                            relative_bounding_box = location.relative_bounding_box
                            rect_start_point = mp_drawing._normalized_to_pixel_coordinates(
                                relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
                                image_rows)
                            rect_end_point = mp_drawing._normalized_to_pixel_coordinates(
                                relative_bounding_box.xmin + relative_bounding_box.width,
                                relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
                                image_rows)
                            if rect_start_point is None or rect_end_point is None:
                                print("rect_points is None.")
                                continue

                            sx, sy = rect_start_point
                            ex, ey = rect_end_point

                            if enable_expand_roi:
                                face_bbox_width = abs(sx - ex)
                                face_bbox_height = abs(sy - ey)
                                sx = round(sx - face_bbox_width * 0.1)
                                ex = round(ex + face_bbox_width * 0.1)
                                sy = round(sy - face_bbox_height * 0.42)
                                ey = round(ey + face_bbox_height * 0.08)
                                
                            sx = max(0, min(image_cols, sx))
                            ex = max(0, min(image_cols, ex))
                            sy = max(0, min(image_rows, sy))
                            ey = max(0, min(image_rows, ey))

                            bbox = list([sx, sy, ex, ey])
                            self.bbox_list.append(bbox)

                        # 処理が終わったタイミングで vis_server へ送信
                        # ref: https://gist.github.com/kittinan/e7ecefddda5616eab2765fdb2affed1b
                        bbox_list_bytes = pickle.dumps(self.bbox_list, 0)
                        size_of_bbox_list = len(bbox_list_bytes)
                        constant_sized_header = struct.pack(
                            ">L", size_of_bbox_list)  # ビックエンディアンで 4byte のサイズ変数を作る
                        # print("send1", len(bbox_list_bytes))
                        self.client_socket_for_vis.sendall(
                            constant_sized_header + bbox_list_bytes)
                        # print("send2", len(bbox_list_bytes))
                        self.client_socket_for_reid.sendall(
                            constant_sized_header + bbox_list_bytes)

                    else:
                        print("result.detections is Nothing.")

                    # 画像の送信（顔検出結果がなくても送る）
                    ret, image = cv2.imencode(".jpg", self.fresh_image, self.encode_param)  # np.array (dim1) へ変換
                    size_of_image = image.shape[0]
                    constant_sized_header2 = struct.pack(">L", size_of_image)  # ビックエンディアンで 4byte のサイズ変数を作る
                    image_bytes = image.tobytes()  # ndarray から純粋なバイト列に変換

                    # ヘッダーをつけてバイト列を送信
                    self.client_socket_for_vis2.sendall(
                        constant_sized_header2 + image_bytes)
                    self.client_socket_for_reid2.sendall(
                        constant_sized_header2 + image_bytes)

                    self.fresh_image = None  # 顔検出が終わったら None にする。

            except Exception as e:
                self.fresh_image = None
                self.bbox_list = list()
                print(e)

            frame_id += 1

        # 止めるときはキルするので下記は実行されない。
        self.client_socket_for_vis.close()
    

    def execute(self):
        # th_for_cam = threading.Thread(target=self.__update_cam_image)
        # th_for_cam.start()
        th_for_face_bbox = threading.Thread(target=self.__update_face_bbox_list)
        th_for_face_bbox.start()


if __name__ == "__main__":
    video_file = None
    try:
        video_file = sys.argv[1]
        print(video_file)
    except:
        pass
    face_det_server = FaceDetServer(video_file)
    face_det_server.execute()
