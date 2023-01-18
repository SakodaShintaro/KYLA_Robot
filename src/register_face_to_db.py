import sqlite3
import cv2
from feature_extractor import FaceRoiExtractor, FaceFeatureExtractor
import numpy as np
from typing import List


def register_feature_into_db(db_path: str, tgt_name: str, data_bytes: bytes) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    try:
        cur.execute(
            'CREATE TABLE persons(id INTEGER PRIMARY KEY AUTOINCREMENT, name STRING, data BLOB)')
        conn.commit()
    except BaseException:
        pass

    # "name"に登録者情報、data に特徴量のバイナリを入れる。
    cur.execute(f'INSERT INTO persons(name, data) values("{tgt_name}", ?)', (memoryview(data_bytes), ))

    conn.commit()
    cur.close()
    conn.close()


def extract_face_region(image: np.ndarray, region: List[float]) -> List[np.ndarray]:
    H, W, _ = image.shape
    image_list = list()
    for bbox in region:
        sx = int(bbox[0] * W)
        sy = int(bbox[1] * H)
        ex = int(bbox[2] * W)
        ey = int(bbox[3] * H)
        cropped_image = image[sy:ey, sx:ex]
        image_list.append(cropped_image)
    return image_list


if __name__ == "__main__":
    name_list = ["hiraike", "kaibara", "kakitsuka", "saito"]

    root_path = f"../assets/sample_images/kyla_members/"
    db_path = f"../assets/database/FACE_FEATURES.db"

    # 顔領域抽出器
    roi_extractor = FaceRoiExtractor(True)

    # 顔の表現推論器
    feature_extractor = FaceFeatureExtractor()

    for name in name_list:
        # 登録画像へのパス
        image_path = f"{root_path}/{name}0.jpg"

        image = cv2.imread(image_path)
        results = roi_extractor.execute(image)

        # 複数人写っている画像を登録には使わないようにする
        assert len(results) == 1, "複数人が登録画像に写っています"

        image_list = extract_face_region(image, results)

        for cropped_image in image_list:
            face_feature = feature_extractor.execute([cropped_image])
            face_feature = face_feature[0]  # 先頭のみを取る
            face_feature_bytes = face_feature.tobytes()
            register_feature_into_db(db_path, name, face_feature_bytes)

        print(f"register {name}")
