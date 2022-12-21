# -*- coding: utf-8 -*-
import torch
import torchvision
from .iresnet import iresnet100
import numpy as np
from typing import List
import os


class FaceFeatureExtractor:
    def __init__(self) -> None:
        self.device_ = torch.device('cuda')
        self.model_ = iresnet100(pretrained=False)
        self.model_.load_state_dict(torch.load(
            f'{os.path.dirname(__file__)}/../../assets/models/backbone.pth',
            map_location=self.device_))
        self.model_.eval()
        self.model_.to(self.device_)

    def execute(self, image_list: List[np.array]):
        image_tensor_list = list()
        for image in image_list:
            image_tensor = torchvision.transforms.functional.to_tensor(image)
            image_tensor = torchvision.transforms.functional.resize(
                size=(112, 112), img=image_tensor)
            image_tensor = torch.unsqueeze(image_tensor, 0)
            image_tensor_list.append(image_tensor)

        torch_cat_image_tensor = torch.cat(image_tensor_list, 0)
        torch_cat_image_tensor = torch_cat_image_tensor.to(self.device_)
        feat_list = self.model_(torch_cat_image_tensor)

        ret_processed_feat_list = list()
        for i in range(len(feat_list)):
            processed_feat = [elem.item() for elem in feat_list[i]]
            ret_processed_feat_list.append(np.array(processed_feat))

        return ret_processed_feat_list
