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
import copy
import traceback

class VisServer(object):
    def __init__(self):
        self.host = 'localhost'
        self.buffer_size = 4096
        self.queue_size = 10
        self.image = None
        self.bbox_list = list()
        self.recognized_name_list = list()

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

        self.port_for_receiving_from_face_reid = 64856
        self.vis_socket_for_face_reid = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        self.vis_socket_for_face_reid.bind(
            (self.host, self.port_for_receiving_from_face_reid))
        self.vis_socket_for_face_reid.listen(self.queue_size)

        self.track_tgt_id_dict = dict()
        self.track_tgt_name_dict = dict()
        self.limit_buffer = 20
        self.cnt_tracks = 0

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
            data_bytes = data[:payload_size]
            data = data[payload_size:]  # 先頭位置をずらす

            image = np.frombuffer(data_bytes, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            self.image = image.copy()

        # 止めるときはキルするので下記は実行されない。
        self.vis_socket_for_cam.close()

    def __iou(self, bbox1, bbox2):
        # ref: https://python-ai-learn.com/2021/02/06/iou/
        # a, bは矩形を表すリストで、a=[xmin, ymin, xmax, ymax]
        ax_mn, ay_mn, ax_mx, ay_mx = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
        bx_mn, by_mn, bx_mx, by_mx = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

        a_area = (ax_mx - ax_mn + 1) * (ay_mx - ay_mn + 1)
        b_area = (bx_mx - bx_mn + 1) * (by_mx - by_mn + 1)

        abx_mn = max(ax_mn, bx_mn)
        aby_mn = max(ay_mn, by_mn)
        abx_mx = min(ax_mx, bx_mx)
        aby_mx = min(ay_mx, by_mx)
        w = max(0, abx_mx - abx_mn + 1)
        h = max(0, aby_mx - aby_mn + 1)
        intersect = w*h

        iou = intersect / (a_area + b_area - intersect)
        return iou

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
            data_bytes = data[:payload_size]
            data = data[payload_size:]  # 先頭位置をずらす
            pre_bbox_list = copy.deepcopy(self.bbox_list[1:])
            self.bbox_list = pickle.loads(
                data_bytes, fix_imports=True, encoding="bytes")
            # self.bbox_frame_id = self.bbox_list[0]

            # Data-Association using IoU
            # self.track_target_dict -> {track0: [1, 0, …], track1: [0, 2, …]} 末尾が最新の BBox-ID
            
            is_associated_bbox_list = [False] * (len(self.bbox_list) - 1)  # self.bbox_list の先頭要素がフレーム ID なので１ずらす
            track_keys = self.track_tgt_id_dict.keys()
            is_tracked_list = [True] * len(track_keys)
            for arr_id, track_token in enumerate(track_keys):
                # e.g. track_token = "track0"
                bbox_id_series = self.track_tgt_id_dict[track_token]

                # 要素が０ならそのオブジェクトは使い終わってるので飛ばす。
                num_bbox_id_elems = len(bbox_id_series)
                if num_bbox_id_elems == 0:
                    is_tracked_list[arr_id] = False
                    continue

                pre_bbox_id = bbox_id_series[-1]
                if pre_bbox_id is None:
                    continue

                pre_bbox = pre_bbox_list[pre_bbox_id]

                max_id = None
                max_iou = -1
                iou_th = 0.4
                for bbox_id, bbox in enumerate(self.bbox_list[1:]):
                    iou = self.__iou(bbox, pre_bbox)
                    if (iou > iou_th) and (max_iou < iou):
                        max_iou = iou
                        max_id = bbox_id

                bbox_id_series.append(max_id)

                if num_bbox_id_elems > self.limit_buffer:
                    bbox_id_series.pop(0)

                if max_id is not None:
                    is_associated_bbox_list[max_id] = True  # 過去の ID と紐づけできたらフラグオン

            # False ならもう使わないので、追跡リストから捨てる。
            unnecessary_keys = [key for key_id, key in enumerate(track_keys) if is_tracked_list[key_id] is False]
            for name in unnecessary_keys:
                self.track_tgt_id_dict.pop(name)
                self.track_tgt_name_dict.pop(name)
                
            # flag が False なら紐づかなかったので、新規オブジェクトを登録
            for bbox_id, flag in enumerate(is_associated_bbox_list):
                if flag is False:
                    self.cnt_tracks = self.cnt_tracks + 1
                    new_track_token = "track" + str(self.cnt_tracks)
                    self.track_tgt_id_dict[new_track_token] = list()
                    new_bbox_id_series = self.track_tgt_id_dict[new_track_token]
                    new_bbox_id_series.append(bbox_id)

                    self.track_tgt_name_dict[new_track_token] = list()
                
        # 止めるときはキルするので下記は実行されない。
        self.vis_socket_for_face_det.close()

    def __update_recog_name_list(self):
        conn_for_face_reid, addr = self.vis_socket_for_face_reid.accept()

        data = b""
        header_size = struct.calcsize(">L")
        while True:
            # ヘッダーをまず読み取って、画像データのサイズを調べる
            while len(data) < header_size:
                data += conn_for_face_reid.recv(self.buffer_size)

            constant_sized_header = data[:header_size]
            payload_size_bytes = constant_sized_header
            data = data[header_size:]  # header 以降がペイロード（の一部。全部拾えてない事があるので）
            payload_size = struct.unpack(">L", payload_size_bytes)[0]

            # 画像サイズ分だけペイロードを少しずつ受け取る。
            # ref: https://gist.github.com/kittinan/e7ecefddda5616eab2765fdb2affed1b
            while len(data) < payload_size:
                recv_data = conn_for_face_reid.recv(self.buffer_size)
                data += recv_data
            data_bytes = data[:payload_size]
            data = data[payload_size:]  # 先頭位置をずらす
            self.recognized_name_list = pickle.loads(
                data_bytes, fix_imports=True, encoding="bytes")
            # self.recog_frame_id = self.recognized_name_list[0]

        # 止めるときはキルするので下記は実行されない。
        self.vis_socket_for_face_reid.close()


    def execute(self):
        # queue = queue.Queue(maxsize=1)
        th_for_cam = threading.Thread(target=self.__update_cam_image)
        th_for_cam.start()
        th_for_face_det = threading.Thread(target=self.__update_face_bbox_list)
        th_for_face_det.start()
        th_for_face_reid = threading.Thread(target=self.__update_recog_name_list)
        th_for_face_reid.start()

        # Comment: kuso code
        while True:
            if self.image is not None:
                len_bbox_list = len(self.bbox_list)
                len_recog_name_list = len(self.recognized_name_list)

                if len_bbox_list > 1 and len_recog_name_list > 1:
                    bbox_frame_id = self.bbox_list[0]
                    recog_frame_id = self.recognized_name_list[0]
                    for bbox_id, bbox in enumerate(self.bbox_list[1:]):
                        sx, sy, ex, ey = bbox
                        bbox_color = (255, 255, 0)
                        thickness = 2
                        cv2.rectangle(self.image, (sx, sy),
                                      (ex, ey), bbox_color, thickness)

                        try:
                            recog_frame_id = self.recognized_name_list[0]
                            recog_bbox_id = None
                            # 今見てる bbox_id と認証時の bbox_id を紐づける。
                            tgt_track_token = "unknown"
                            for track_token in self.track_tgt_id_dict.keys():
                                bbox_id_series = self.track_tgt_id_dict[track_token]
                                # 今見てる bbox_id と末尾の ID が一致するか調べる。
                                if bbox_id == bbox_id_series[-1]:
                                    ref_id = (bbox_frame_id - recog_frame_id) * -1
                                    try:
                                        recog_bbox_id = bbox_id_series[ref_id - 1]  # 末尾が最新フレームなので、フレーム ID の差分から参照する。
                                    except:
                                        pass
                                    tgt_track_token = track_token
                                    break
                            if tgt_track_token == "unknown":
                                continue
                            tgt_text = "unknown"
                            if recog_bbox_id is not None:
                                try:
                                    tgt_text = self.recognized_name_list[recog_bbox_id + 1]  # 先頭要素はフレーム ID なので１ずらして参照する。
                                except:
                                    pass

                            max_cnt = 0
                            self.track_tgt_name_dict[tgt_track_token].append(tgt_text)

                            bbox_name_series = self.track_tgt_name_dict[track_token]   # 統計処理で使う
                            num_bbox_name_elems = len(bbox_name_series)
                            if num_bbox_name_elems > self.limit_buffer:
                                bbox_name_series.pop(0)

                            track_tgt_name_set = list(set(self.track_tgt_name_dict[tgt_track_token]))

                            # ある程度バッファに情報がたまっているなら統計処理を行う。
                            # ある程度たまっていないと誤検知が起こる。
                            if num_bbox_name_elems > self.limit_buffer / 2:
                                for tgt_name in track_tgt_name_set:
                                    if tgt_name is None:
                                        continue
                                    cnt = self.track_tgt_name_dict[tgt_track_token].count(tgt_name)
                                    if max_cnt < cnt:
                                        max_cnt = cnt
                                        tgt_text = tgt_name

                            if tgt_text == "unknown":
                                name_color = (0, 0, 255)
                            else:
                                name_color = bbox_color
                            cv2.putText(self.image, tgt_text + ", id: " + str(bbox_id) + ", " + tgt_track_token,
                                        (sx, sy - 15), 1, 1.5, name_color, thickness)
                        except:
                            print(recog_frame_id, bbox_frame_id)
                            traceback.print_exc()
                            pass

                cv2.imshow('ImageWindow', self.image)
                delay = 1
                key = cv2.waitKey(delay)
                if key == 27:
                    cv2.destroyAllWindows()
                    break


if __name__ == "__main__":
    vis_server = VisServer()
    vis_server.execute()
