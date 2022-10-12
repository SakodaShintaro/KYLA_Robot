# -*- coding: utf-8 -*-
import cv2
import glob
import os
import mediapipe as mp
import numpy as np
from typing import List


class FaceExtractor:
    def __init__(self, enable_expand_roi: bool) -> None:
        mp_face_detection = mp.solutions.face_detection
        self.face_defector_ = mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)
        self.mp_drawing_ = mp.solutions.drawing_utils
        self.enable_expand_roi_ = enable_expand_roi

    def execute(self, image: np.array) -> List[List[float]]:
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = self.face_defector_.process(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        result = list()

        image_rows, image_cols, _ = image.shape

        for detection in results.detections:
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
            
            sx /= image_cols
            sy /= image_rows
            ex /= image_cols
            ey /= image_rows

            result.append([sx, sy, ex, ey])

        return result


if __name__ == "__main__":
    image_path_list = glob.glob("raw_data/*.*")
    image_path_list = sorted(image_path_list)

    extractor = FaceExtractor(True)

    save_dir = "only_face_data"
    os.makedirs(save_dir, exist_ok=True)

    for idx, file in enumerate(image_path_list):
        print(file)
        image = cv2.imread(file)
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        result = extractor.execute(image)

        roi_image = image.copy()
        image_rows, image_cols, _ = image.shape
        for detection in result:
            sx, sy, ex, ey = detection
            roi_image = roi_image[sy:ey + 1, sx:ex + 1, :]
            break  # 一個しかないので抜ける

        file_basename = os.path.basename(file)
        cv2.imwrite(f'{save_dir}/{file_basename}', roi_image)
