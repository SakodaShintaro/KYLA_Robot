import sqlite3
import cv2
import argparse
from feature_extractor import FaceRoiExtractor, FaceFeatureExtractor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("image_path", type=str)
    return parser.parse_args()


def register_feature_into_db(tgt_name, data_bytes):
    dbname = 'FACE_FEATURES.db'
    conn = sqlite3.connect(dbname)
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


if __name__ == "__main__":
    args = get_args()
    tgt_name = args.name  # 登録者名
    tgt_image_path = args.image_path  # 登録画像（顔画像を含む）

    # 顔領域抽出器
    roi_extractor = FaceRoiExtractor(True)
    image = cv2.imread(tgt_image_path)
    results = roi_extractor.execute(image)

    # 顔の表現推論器
    feature_extractor = FaceFeatureExtractor()

    # 複数人写っている画像を登録には使わないようにする
    assert len(results) == 1, "複数人が登録画像に写っています"

    H, W, _ = image.shape

    for bbox in results:
        sx = int(bbox[0] * W)
        sy = int(bbox[1] * H)
        ex = int(bbox[2] * W)
        ey = int(bbox[3] * H)
        cropped_image = image[sy:ey, sx:ex]

        face_feature = feature_extractor.execute([cropped_image])
        face_feature = face_feature[0]  # 先頭のみを取る
        face_feature_bytes = face_feature.tobytes()
        register_feature_into_db(tgt_name, face_feature_bytes)
