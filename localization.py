from backbone.model import VGG16FeatureExtractor
from common import *
import os
from skimage.transform import resize

import torch
import torch.nn as nn


class LocationDetector:
    def __init__(self, feature_model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.__feature_model = feature_model
        self.__device = device
        self.__avg_poolinger = nn.AvgPool2d(kernel_size=2, stride=2)

    def __global_am_pooling(self, feature_map):
        channel_size = feature_map.shape[0]
        max_vector = torch.tensor([feature_map[i, :, :].max() for i in range(channel_size)], device=self.__device)
        avg_vector = torch.tensor([feature_map[i, :, :].mean() for i in range(channel_size)], device=self.__device)
        return torch.cat([avg_vector, max_vector], dim=0)

    def __region_amac(self, feature_map):
        pass

    def __score_map(self, target_feature, query_feature, upsamp_size):
        conv_map = torch.nn.functional.conv2d(target_feature, query_feature)
        data = conv_map.numpy()
        up_data = resize(data, upsamp_size)
        return torch.tensor(up_data, device=self.__device)

    def detect_location(self, target_img, query_img):
        target_tensor = torch.tensor(target_img / 255, dtype=torch.double, device=self.__device)
        query_tensor = torch.tensor(query_img / 255, dtype=torch.double, device=self.__device)
        target_conv_features, target_max_features = self.__feature_model.representations_of(target_tensor)
        target_avg_features = [self.__avg_poolinger(feature) for feature in target_conv_features]
        # A&MP
        target_fusion_features = [torch.cat([target_avg_features[i], target_max_features[i]], dim=0)
                                  for i in range(len(target_avg_features))]

        query_conv_features, query_max_features = self.__feature_model.representations_of(query_tensor)
        # GA&MP
        query_fusion_features = [self.__global_am_pooling(query_conv_features[i]) for i in range(3)]
        # R-AMAC
        query_fusion_features += [self.__region_amac(query_conv_features[i]) for i in range(4, 6)]
        # score maps
        score_maps = [self.__score_map(target_fusion_features[i], query_fusion_features[i],
                                       (target_img.shape[0], target_img.shape[1]))
                      for i in range(len(target_fusion_features))]
        fusion_map = torch.mean(torch.tensor(score_maps), dim=0)
        # TODO 2stage detection


if __name__ == '__main__':
    model_filename = 'vgg16_bn-6c64b313.pth'
    model_file_path = os.path.join(proj_path, 'model_zoo', 'checkpoints', model_filename)
