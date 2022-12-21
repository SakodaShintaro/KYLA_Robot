# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from feature_extractor import FaceRoiExtractor
from feature_extractor import FaceFeatureExtractor
import glob
import os
import numpy as np
import cv2


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def vis_arcface():
    # 顔領域抽出器生成
    face_roi_extractor = FaceRoiExtractor(enable_expand_roi=True)

    # 特徴量計算器生成
    face_feature_extractor = FaceFeatureExtractor()

    # 画像読み込み & 顔領域抽出
    image_path_list = sorted(glob.glob("../assets/sample_images/kyla_members/*.jpg"))
    image_list = list()
    for image_path in image_path_list:
        image = cv2.imread(image_path)
        roi_list = face_roi_extractor.execute(image)
        h, w, _ = image.shape
        for roi in roi_list:
            sx, sy, ex, ey = roi
            sx = int(w * sx)
            sy = int(h * sy)
            ex = int(w * ey)
            ey = int(h * ey)
            image_list.append(image[sy:ey, sx:ex])

    # 抽出した画像を保存
    save_dir = "extracted_face_image"
    os.makedirs(save_dir, exist_ok=True)
    for i, image in enumerate(image_list):
        cv2.imwrite(f"{save_dir}/{i:06}.png", image)

    # 抽出した画像について推論
    processed_feat_list = face_feature_extractor.execute(image_list)

    # 顔特徴量比較
    save_dir = "vis_graph"
    os.makedirs(save_dir, exist_ok=True)

    for anchor_id in range(len(processed_feat_list)):
        anchor_feat = [elem.item() for elem in processed_feat_list[anchor_id]]

        plot_value_list = list()
        for i in range(len(processed_feat_list)):
            processed_feat = processed_feat_list[i]
            cos_sim_value = cos_sim(np.array(anchor_feat),
                                    np.array(processed_feat))
            plot_value_list.append(cos_sim_value)

        # FigureとAxesを作成
        _, ax = plt.subplots()
        plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_xlabel("image id")
        ax.set_ylabel("cosine similarity")

        # Axesに縦棒グラフを追加
        left = np.array(np.array(range(len(image_path_list))))
        height = np.array(plot_value_list)
        ax.bar(left, height, width=0.9, align="center", color="blue")

        only_anchor_left = np.array(anchor_id)
        only_anchor_height = np.array(plot_value_list[anchor_id])
        ax.bar(only_anchor_left, only_anchor_height, width=0.9, align="center", color="red")

        plt.savefig(f"{save_dir}/anchor{anchor_id:02d}.png")

    print("finish")


if __name__ == "__main__":
    vis_arcface()
