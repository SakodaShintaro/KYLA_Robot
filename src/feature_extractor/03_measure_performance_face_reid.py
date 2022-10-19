# -*- coding: utf-8 -*-
from tkinter import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import normal_feature_extractor
import arcface_feature_extractor
import glob
import os
import numpy as np


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def vis_arcface():
    # image_path_list = glob.glob("raw_data/*.*")
    # image_path_list = glob.glob("only_face_data/*.*")
    image_path_list = glob.glob("only_face_data_kyla_members/*.*")

    processed_feat_list = arcface_feature_extractor.execute(image_path_list)

    # if not os.path.exists("vis_graph"):
    #     os.mkdir("vis_graph")

    # for anchor_id in range(len(processed_feat_list)):
    for anchor_id in range(len(image_path_list)):
        image_path = image_path_list[anchor_id]
        processed_feat_for_perf = arcface_feature_extractor.execute([image_path])

        anchor_feat = [elem.item() for elem in processed_feat_for_perf[0]]
        # anchor_image_basename = os.path.basename(image_path_list[anchor_id])
        # print("anchor:", anchor_image_basename)
        continue

        plot_value_list = list()
        for i in range(len(processed_feat_list)):
            processed_feat = processed_feat_list[i]
            cos_sim_value = cos_sim(np.array(anchor_feat),
                                    np.array(processed_feat))
            # image_basename = os.path.basename(image_path_list[i])
            # print("id-" + str(i) + ",", image_basename + ",",
            #       cos_sim_value)
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

        # plt.show()
        # plt.savefig("vis_graph/anchor" + str(anchor_id) + ".png")
        plt.savefig("vis_graph/anchor" + str(anchor_id) + "_" + os.path.basename(image_path_list[anchor_id]) + ".png")


if __name__ == "__main__":
    vis_arcface()
