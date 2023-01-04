import cv2
from feature_extractor import FaceRoiExtractor, FaceMatcher
from register_face_to_db import extract_face_region
from glob import glob


if __name__ == "__main__":
    # 顔一致器
    dbname = 'FACE_FEATURES.db'
    face_matcher = FaceMatcher(dbname)

    # 顔領域抽出器
    roi_extractor = FaceRoiExtractor(True)

    image_path_list = sorted(glob(f"../assets/sample_images/kyla_members/*.jpg"))
    image_list = list()
    for image_path in image_path_list:
        image = cv2.imread(image_path)
        regions = roi_extractor.execute(image)
        cropped_image_list = extract_face_region(image, regions)
        image_list.append(cropped_image_list[0])
    
    result = face_matcher(image_list)

    for sim, name in result:
        print(f"sim = {sim:.4f}, name = {name}")
