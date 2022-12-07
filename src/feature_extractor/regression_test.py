# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import arcface_feature_extractor
import glob
import os
import numpy as np


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def vis_arcface():
    image_path_list = sorted(glob.glob("../../assets/sample_images/kyla_members/*.jpg"))

    processed_feat_list = arcface_feature_extractor.execute(image_path_list)

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
