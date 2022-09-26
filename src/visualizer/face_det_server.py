# -*- coding: utf-8 -*-
# ref: https://gist.github.com/kittinan/e7ecefddda5616eab2765fdb2affed1b
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
        self.port_for_receiving_from_cam = 64850
        self.buffer_size = 4096 * 4
        self.queue_size = 10
        self.fresh_image = None
        self.bbox_list = list()
        self.stop_threads = False

        self.port_for_sending_to_vis = 64852
        self.client_socket_for_vis = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket_for_vis.connect(('localhost', self.port_for_sending_to_vis))

    def __update_face_bbox_list(self):
        while not self.stop_threads:
            if self.fresh_image is None:
                print("fresh_image is None.")
                continue

            enable_expand_roi = True

            try:
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
                        data = pickle.dumps(self.bbox_list, 0)
                        size = len(data)
                        self.client_socket_for_vis.sendall(struct.pack(">L", size) + data)
                        self.fresh_image = None
                    else:
                        # cv2.imwrite("temp.png", self.fresh_image)
                        self.fresh_image = None
                        print("result.detections is Nothing.")

            except Exception as e:
                self.fresh_image = None
                self.bbox_list = list()
                print(e)

    def execute(self):
        # queue = queue.Queue(maxsize=1)
        th = threading.Thread(target=self.__update_face_bbox_list)
        th.start()

        face_det_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('Socket created')

        face_det_socket.bind((self.host, self.port_for_receiving_from_cam))
        print('Socket bind complete')
        face_det_socket.listen(self.queue_size)
        print('Socket now listening')

        conn, addr = face_det_socket.accept()

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
            # for arr_id, bbox in enumerate(self.bbox_list):
            #     sx, sy, ex, ey = bbox
            #     color = (255, 255, 0)
            #     thickness = 2
            #     cv2.rectangle(frame, (sx, sy), (ex, ey), color, thickness)
            #     cv2.putText(frame, "face_id: " + str(arr_id), (sx, sy - 15), 1, 1.5, color, thickness)
            # cv2.imshow('ImageWindow', frame)
            # key = cv2.waitKey(1)
            # if key == 27:
            #     cv2.destroyAllWindows()
            #     self.stop_threads = True
            #     break

        face_det_socket.close()


if __name__ == "__main__":
    face_det_server = FaceDetServer()
    face_det_server.execute()
