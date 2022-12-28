import sqlite3
import numpy as np
import cv2
from feature_extractor import FaceRoiExtractor, FaceFeatureExtractor
from register_face_to_db import extract_face_region
from glob import glob
import matplotlib.pyplot as plt


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

        image_path_list = sorted(glob(f"../assets/sample_images/kyla_members/*.jpg"))

        compare_image_list = list()

        for image_path in image_path_list:
            image = cv2.imread(image_path)
            regions = roi_extractor.execute(image)
            image_list = extract_face_region(image, regions)
            compare_image_list.append(image_list[0])

        feat_list = feature_extractor.execute(compare_image_list)

        sim_list = [cos_sim(face_feature, f) for f in feat_list]

        plt.bar([i for i in range(len(sim_list))], sim_list)
        plt.xlabel("Image ID")
        plt.ylabel("Similarity")

        save_path = f"verify_db_{name}.png"
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
        print(f"save to {save_path}")
        plt.close()
