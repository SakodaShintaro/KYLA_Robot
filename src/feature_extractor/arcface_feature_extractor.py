# -*- coding: utf-8 -*-
import torch
import torchvision
from PIL import Image
from iresnet import iresnet100
import numpy as np


def execute(image_path_list):
    device = torch.device('cpu')
    model = iresnet100(pretrained=False)

    # ref: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch#model-zoo
    model.load_state_dict(torch.load(
        'models/backbone_r100.pth', map_location=device))

    model.eval()
    # image_path_list = glob.glob("only_face_data/*.*")
    image_tensor_list = list()
    for image_path in image_path_list:
        image = Image.open(image_path)
        image = image.convert("RGB")
        image_tensor = torchvision.transforms.functional.to_tensor(image)
        image_tensor = torchvision.transforms.functional.resize(
            size=(112, 112), img=image_tensor)
        image_tensor = torch.unsqueeze(image_tensor, 0)
        image_tensor_list.append(image_tensor)

    torch_cat_mage_tensor = torch.cat(image_tensor_list, 0)
    outputs = model(torch_cat_mage_tensor)
    feat_list = outputs

    ret_processed_feats = list()
    for i in range(len(feat_list)):
        processed_feat = [elem.item() for elem in feat_list[i]]
        ret_processed_feats.append(np.array(processed_feat))

    return ret_processed_feats
