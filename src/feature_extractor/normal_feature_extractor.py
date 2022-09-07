from timm.models.efficientnet import EfficientNet
import torch
import torchvision
import torch.nn.functional as F
import timm
from PIL import Image
import numpy as np


def extract_features(inputs: torch.Tensor, model: EfficientNet):
    with torch.no_grad():
        features = dict()
        # extract stem features as level 1
        x = model.conv_stem(inputs)
        x = model.bn1(x)
        x = model.bn1.act(x)
        # x = model.act1(x)
        features['level_1'] = F.adaptive_avg_pool2d(x, 1)
        # extract blocks features as level 2~8
        for i, block_layer in enumerate(model.blocks):
            x = block_layer(x)
            features[f'level_{i+2}'] = F.adaptive_avg_pool2d(x, 1)
        # extract top features as level
        x = model.conv_head(x)
        x = model.bn2(x)
        x = model.bn2.act(x)
        # x = model.act2(x)
        features['level_9'] = F.adaptive_avg_pool2d(x, 1)
        return features


def extract_feature(inputs, model):
    features = extract_features(inputs, model)
    return features["level_9"]


def execute(image_path_list):
    model = timm.create_model('tf_efficientnet_b4_ns', pretrained=True)

    model.eval()
    model.to('cpu')
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
    feat_list = extract_feature(torch_cat_mage_tensor, model)

    ret_processed_feat_list = list()
    for i in range(len(feat_list)):
        processed_feat = [elem[0][0].item() for elem in feat_list[i]]
        ret_processed_feat_list.append(np.array(processed_feat))

    return ret_processed_feat_list
