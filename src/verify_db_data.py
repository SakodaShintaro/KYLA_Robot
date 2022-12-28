import sqlite3
import numpy as np
import cv2
from feature_extractor import FaceRoiExtractor, FaceFeatureExtractor
from register_face_to_db import extract_face_region


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


if __name__ == "__main__":
    # sqltie3 ---
    dbname = 'FACE_FEATURES.db'
    conn = sqlite3.connect(dbname)
    cur = conn.cursor()

    # 顔領域抽出器
    roi_extractor = FaceRoiExtractor(True)

    # 顔の表現推論器
    feature_extractor = FaceFeatureExtractor()

    # terminalで実行したSQL文と同じようにexecute()に書く
    cur.execute('SELECT * FROM persons')
    raw_registered_table_data = cur.fetchall()
    # バイナリから元の情報へ戻す。
    for tuple_data in raw_registered_table_data:
        register_id, name, face_feature_bytes = tuple_data
        print(name)
        face_feature = np.frombuffer(face_feature_bytes, dtype=np.float)

        image_path0 = f"../assets/sample_images/kyla_members/{name}0.jpg"
        image0 = cv2.imread(image_path0)
        image_path1 = f"../assets/sample_images/kyla_members/{name}1.jpg"
        image1 = cv2.imread(image_path1)

        regions0 = roi_extractor.execute(image0)
        regions1 = roi_extractor.execute(image1)

        image_list0 = extract_face_region(image0, regions0)
        image_list1 = extract_face_region(image1, regions1)

        feat_list = feature_extractor.execute([image_list0[0], image_list1[0]])
        sim0 = cos_sim(face_feature, feat_list[0])
        sim1 = cos_sim(face_feature, feat_list[1])
        print(f"{sim0}, {sim1}")
