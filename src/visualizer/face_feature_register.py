import sqlite3
import numpy as np
import sys
import torch
import torchvision
from PIL import Image
from iresnet import iresnet100
import mediapipe as mp
from typing import List
import cv2


class FaceExtractor:
    """
        Comment: 共通化したい。
    """
    def __init__(self, enable_expand_roi: bool) -> None:
        mp_face_detection = mp.solutions.face_detection
        self.face_defector_ = mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)
        self.mp_drawing_ = mp.solutions.drawing_utils
        self.enable_expand_roi_ = enable_expand_roi

    def execute(self, image: np.array) -> List[List[float]]:
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        mp_results = self.face_defector_.process(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        results = list()

        image_rows, image_cols, _ = image.shape

        for detection in mp_results.detections:
            location = detection.location_data
            keypoint_list = location.relative_keypoints

            relative_bounding_box = location.relative_bounding_box
            rect_start_point = self.mp_drawing_._normalized_to_pixel_coordinates(
                max(relative_bounding_box.xmin, 0),
                relative_bounding_box.ymin,
                image_cols,
                image_rows)
            rect_end_point = self.mp_drawing_._normalized_to_pixel_coordinates(
                min(relative_bounding_box.xmin + relative_bounding_box.width, 1),
                min(relative_bounding_box.ymin + relative_bounding_box.height, 1),
                image_cols,
                image_rows)

            sx, sy = rect_start_point
            ex, ey = rect_end_point

            if self.enable_expand_roi_:
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
            
            results.append([sx, sy, ex, ey])

        return results

class FaceFeatureRegister(object):
    def __init__(self):
        # IResNet config ---
        # self.device = torch.device('cuda')
        self.device = torch.device('cpu')
        self.model = iresnet100(pretrained=False)

        # ref: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch#model-zoo
        self.model.load_state_dict(torch.load(
            '../models/backbone.pth',
            map_location=self.device))
        # --- IResNet config
    
    def __cv2pil(self, image):
        ''' OpenCV型 -> PIL型 
        Comments:
            共通化したい。
        '''
        new_image = image.copy()
        if new_image.ndim == 2:  # モノクロ
            pass
        elif new_image.shape[2] == 3:  # カラー
            new_image = new_image[:, :, ::-1]
        elif new_image.shape[2] == 4:  # 透過
            new_image = new_image[:, :, [2, 1, 0, 3]]
        new_image = Image.fromarray(new_image)
        return new_image

    def register_feature_into_db(self, tgt_name, data_bytes):
        dbname = 'FACE_FEATURES.db'
        conn = sqlite3.connect(dbname)
        cur = conn.cursor()

        try:
            cur.execute(
                'CREATE TABLE persons(id INTEGER PRIMARY KEY AUTOINCREMENT, name STRING, data BLOB)')
                # 'CREATE TABLE persons(id INTEGER PRIMARY KEY AUTOINCREMENT, name STRING, data INTEGER)')
            conn.commit()
        except:
            pass

        # "name"に登録者情報、data に特徴量のバイナリを入れる。
        cur.execute('INSERT INTO persons(name, data) values("{}", ?)'.format(tgt_name), (memoryview(data_bytes), ))

        conn.commit()
        cur.close()
        conn.close()

    def extract_reid_feature(self, image_np):
        """
            Comment: 共通化したい。
        """
        self.model.eval()
        self.model.to(self.device)
        image_tensor_list = list()

        image = self.__cv2pil(image_np)
        image = image.convert("RGB")
        image_tensor = torchvision.transforms.functional.to_tensor(image)
        image_tensor = torchvision.transforms.functional.resize(
            size=(112, 112), img=image_tensor)
        image_tensor = torch.unsqueeze(image_tensor, 0)
        image_tensor_list.append(image_tensor)

        torch_cat_image_tensor = torch.cat(image_tensor_list, 0)
        torch_cat_image_tensor = torch_cat_image_tensor.to(self.device)
        feat_list = self.model(torch_cat_image_tensor)

        # ret_processed_feat_list = list()
        # for i in range(len(feat_list)):
        processed_feat = [elem.item() for elem in feat_list[0]]
        ret_processed_feat = np.array(processed_feat)

        # print(feat_list[0].shape)
        return ret_processed_feat

if __name__ == "__main__":
    tgt_name = sys.argv[1]  # 登録者名
    tgt_image_path = sys.argv[2]  # 登録画像（顔画像を含む）

    extractor = FaceExtractor(True)
    image = cv2.imread(tgt_image_path)
    results = extractor.execute(image)

    face_feature_register = FaceFeatureRegister()

    cropped_image_list = list()
    for bbox in results:
        sx = bbox[0]
        sy = bbox[1]
        ex = bbox[2]
        ey = bbox[3]
        # print(bbox)
        cropped_image = image[sy:ey, sx:ex]
        cv2.imwrite("croppted_fresh_image.png", cropped_image)
        cropped_image_list.append(cropped_image)

        face_feature = face_feature_register.extract_reid_feature(cropped_image)
        # print(face_feature.dtype)
        # exit()
        face_feature_bytes = face_feature.tobytes()
        # print(face_feature_bytes)
        face_feature_register.register_feature_into_db(tgt_name, face_feature_bytes)
        break # まだ一人ずつしか登録できないが、複数人一度に登録できるようにしてもいいかも。

    # face_feature_bytes = face_feature_register.extract_reid_feature(image_np)


    
    