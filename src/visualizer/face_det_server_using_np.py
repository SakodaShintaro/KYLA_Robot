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


class FaceDetServer(object):
    def __init__(self):
        self.host = 'localhost'
        self.port = 64850
        self.buffer_size = 4096 * 4
        self.queue_size = 10
        self.fresh_image = None
        self.bbox_list = list()

    def __update_face_bbox_list(self):
        while True:
            if self.fresh_image is None:
                continue

            enable_expand_roi = True

            mp_face_detection = mp.solutions.face_detection
            mp_drawing = mp.solutions.drawing_utils

            with mp_face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.5) as face_detection:
                # image = cv2.imread(file)
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

                # file_basename = os.path.basename(file)
                # cv2.imwrite('only_face_data/' + file_basename, roi_image)
            self.fresh_image = None
            # print("Done face detection.")  # face's key-points
            # return ret_bbox_list

    def execute(self):
        # queue = queue.Queue(maxsize=1)
        th = threading.Thread(target=self.__update_face_bbox_list)
        th.start()

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('Socket created')

        s.bind((self.host, self.port))
        print('Socket bind complete')
        s.listen(self.queue_size)
        print('Socket now listening')

        conn, addr = s.accept()

        data = b""
        header_size = struct.calcsize(">L")
        while True:
            while len(data) < header_size:
                data += conn.recv(self.buffer_size)

            constant_sized_header = data[:header_size]
            payload_size = constant_sized_header
            data = data[header_size:]  # header 以降がペイロード（の一部。全部拾えてない事があるので）
            msg_size = struct.unpack(">L", payload_size)[0]
            # この時点ではヘッダーは拾えているが、ペイロードはまだ受け取れていない。

            # ペイロードを少しずつ受け取る。
            while len(data) < msg_size:
                recv_data = conn.recv(self.buffer_size)
                data += recv_data
            frame_data = data[:msg_size]
            data = data[msg_size:]  # リセット

            frame = np.fromstring(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

            # self.fresh_image は face-detection が完了すると None になり、以下で更新される。
            # なので、face-detection には常に新鮮な画像が入力される。
            if self.fresh_image is None:
                self.fresh_image = frame.copy()

            print(self.bbox_list)    
            for arr_id, bbox in enumerate(self.bbox_list):
                sx, sy, ex, ey = bbox
                color = (255, 255, 255)
                thickness = 2
                cv2.rectangle(frame, (sx, sy), (ex, ey), color, thickness)
                cv2.putText(frame, "face_id: " + str(arr_id), (sx, sy - 15), 1, 2.0, color, thickness)
            cv2.imshow('ImageWindow', frame)
            cv2.waitKey(1)


if __name__ == "__main__":
    face_det_server = FaceDetServer()
    face_det_server.execute()
