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
    image_path_list = glob.glob("raw_data/*.*")
    # image_path_list = glob.glob("only_face_data/*.*")
    processed_feat_list = arcface_feature_extractor.execute(image_path_list)

    if not os.path.exists("vis_graph"):
        os.mkdir("vis_graph")

    for anchor_id in range(len(processed_feat_list)):
        anchor_feat = [elem.item() for elem in processed_feat_list[anchor_id]]
        # anchor_image_basename = os.path.basename(image_path_list[anchor_id])
        # print("anchor:", anchor_image_basename)

        plot_value_list = list()
        for i in range(len(processed_feat_list)):
            processed_feat = processed_feat_list[i]
            cos_sim_value = cos_sim(np.array(anchor_feat),
                                    np.array(processed_feat))
            # image_basename = os.path.basename(image_path_list[i])
            # print("id-" + str(i) + ",", image_basename + ",",
            #       cos_sim_value)
            plot_value_list.append(cos_sim_value)

        left = np.array(np.array(range(len(image_path_list))))
        height = np.array(plot_value_list)
        # FigureとAxesを作成
        _, ax = plt.subplots()
        plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_xlabel("image id")
        ax.set_ylabel("cosine similarity")

        # Axesに縦棒グラフを追加
        ax.bar(left, height, width=0.9, align="center")

        # plt.show()
        plt.savefig("vis_graph/anchor" + str(anchor_id) + ".png")


if __name__ == "__main__":
    vis_arcface()