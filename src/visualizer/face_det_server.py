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
    def __init__(self):
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

        print('Setting Sockets')

    def __update_cam_image(self):
        conn_for_cam, addr = self.vis_socket_for_cam.accept()

        data = b""
        header_size = struct.calcsize(">L")
        while True:
            while len(data) < header_size:
                data += conn_for_cam.recv(self.buffer_size)

            constant_sized_header = data[:header_size]
            payload_size = constant_sized_header
            data = data[header_size:]  # header 以降がペイロード（の一部。全部拾えてない事があるので）
            msg_size = struct.unpack(">L", payload_size)[0]
            # この時点ではヘッダーは拾えているが、ペイロードはまだ受け取れていない。

            # ペイロードを少しずつ受け取る。
            # ref: https://gist.github.com/kittinan/e7ecefddda5616eab2765fdb2affed1b
            while len(data) < msg_size:
                recv_data = conn_for_cam.recv(self.buffer_size)
                data += recv_data
            frame_data = data[:msg_size]
            data = data[msg_size:]  # リセット

            image = np.fromstring(frame_data, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            self.fresh_image = image.copy()

        self.vis_socket_for_cam.close()

    def __update_face_bbox_list(self):
        # face_det_cnt = 0
        while True:
            if self.fresh_image is None:
                print("fresh_image is None.")
                continue

            enable_expand_roi = True
            try:
                mp_face_detection = mp.solutions.face_detection
                mp_drawing = mp.solutions.drawing_utils

                with mp_face_detection.FaceDetection(
                        model_selection=1, min_detection_confidence=0.5) as face_detection:
                    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
                    results = face_detection.process(
                        cv2.cvtColor(self.fresh_image, cv2.COLOR_BGR2RGB))

                    # Draw face detections of each face.
                    if results.detections:
                        self.bbox_list = list()
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

                            bbox = list([sx, sy, ex, ey])
                            self.bbox_list.append(bbox)

                        # 処理が終わったタイミングで vis_server へ送信
                        # ref: https://gist.github.com/kittinan/e7ecefddda5616eab2765fdb2affed1b
                        data = pickle.dumps(self.bbox_list, 0)
                        size = len(data)
                        self.client_socket_for_vis.sendall(
                            struct.pack(">L", size) + data)
                        self.fresh_image = None
                    else:
                        self.fresh_image = None
                        print("result.detections is Nothing.")

            except Exception as e:
                self.fresh_image = None
                self.bbox_list = list()
                print(e)

        self.client_socket_for_vis.close()

    def execute(self):
        th_for_cam = threading.Thread(target=self.__update_cam_image)
        th_for_cam.start()
        th_for_face_det = threading.Thread(target=self.__update_face_bbox_list)
        th_for_face_det.start()


if __name__ == "__main__":
    face_det_server = FaceDetServer()
    face_det_server.execute()
