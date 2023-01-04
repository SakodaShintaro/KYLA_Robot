# -*- coding: utf-8 -*-
""" データベースに登録した顔画像と比較して名前を返すクラス
"""

import numpy as np
from typing import List
import sqlite3
from .face_feature_extractor import FaceFeatureExtractor
from .face_roi_extractor import FaceRoiExtractor


class FaceMatcher:
    def __init__(self, db_path: str) -> None:
        # 顔領域抽出器
        self.roi_extractor = FaceRoiExtractor(True)

        # 顔の表現推論器
        self.feature_extractor = FaceFeatureExtractor()

        # DBの準備
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute('SELECT * FROM persons')
        self.raw_registered_table_data = cur.fetchall()

    def __call__(self, image_list: List[np.ndarray]) -> List[str]:
        feat_list = self.feature_extractor.execute(image_list)

        result = list()
        for f in feat_list:
            max_sim = -float("inf")
            max_name = "None"
            for tuple_data in self.raw_registered_table_data:
                register_id, name, face_feature_bytes = tuple_data
                face_feature = np.frombuffer(face_feature_bytes, dtype=np.float)
                curr_sim = self.cos_sim(face_feature, f)
                if curr_sim > max_sim:
                    max_sim = curr_sim
                    max_name = name
            result.append((max_sim, max_name))

        return result

    def cos_sim(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
