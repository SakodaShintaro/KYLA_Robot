# -*- coding: utf-8 -*-
import cv2
import glob
import os
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

image_path_list = glob.glob("raw_data/*.*")

if not os.path.exists("only_face_data"):
    os.mkdir("only_face_data")

with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
    for idx, file in enumerate(image_path_list):
        image = cv2.imread(file)
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw face detections of each face.
        if not results.detections:
            continue

        roi_image = image.copy()
        image_rows, image_cols, _ = image.shape
        for detection in results.detections:
            location = detection.location_data
            relative_bounding_box = location.relative_bounding_box
            rect_start_point = mp_drawing._normalized_to_pixel_coordinates(
                relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
                image_rows)
            rect_end_point = mp_drawing._normalized_to_pixel_coordinates(
                relative_bounding_box.xmin + relative_bounding_box.width,
                relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
                image_rows)

            # print('Nose tip:')
            # print(mp_face_detection.get_key_point(
            #     detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
            # mp_drawing.draw_detection(annotated_image, detection)

            sx, sy = rect_start_point
            ex, ey = rect_end_point
            roi_image = roi_image[sy:ey+1, sx:ex+1, :]

            break  # 一個しかないので抜ける

        file_basename = os.path.basename(file)
        cv2.imwrite('only_face_data/' + file_basename, roi_image)
