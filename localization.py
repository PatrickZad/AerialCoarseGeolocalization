from backbone.model import VGG16FeatureExtractor
from common import *
import os
from skimage.transform import resize
from skimage.io import imsave, imread

from backbone.model import VGG16FeatureExtractor
import torch
import torch.nn as nn

import numpy as np


class LocationDetector:
    def __init__(self, feature_model, device='cuda' if torch.cuda.is_available() else 'cpu'):

        self.__feature_model = VGG16FeatureExtractor(vgg16_file=feature_model, device=device)
        self.__device = device
        self.__min_scale = 8
        self.__min_region = 2
        # self.__avg_poolinger = nn.AvgPool2d(kernel_size=2, stride=2)

    def __global_am_pooling(self, feature_map):
        # max_map = torch.max(feature_map, dim=-1)
        # max_vector = torch.max(max_map, dim=-1)
        max_map=feature_map.max(dim=-1)[0]
        max_vector=max_map.max(dim=-1)[0]
        sum_map = torch.sum(feature_map, dim=-1)
        sum_map = torch.sum(sum_map, dim=-1)
        avg_vector = sum_map / (feature_map.shape[-1] * feature_map.shape[-2])
        '''channel_size = feature_map.shape[0]
        max_vector = torch.tensor([feature_map[i, :, :].max() 
                                   for i in range(channel_size)], device=self.__device)
        avg_vector = torch.tensor([feature_map[i, :, :].mean() 
                                   for i in range(channel_size)], device=self.__device)'''
        return torch.cat([avg_vector, max_vector], dim=0)

    def __region_amac(self, feature_map):
        c, h, w = feature_map.shape[1:]  # feature_map.shape[1], feature_map.shape[2]
        max_r = min(h, w)
        if h == w:
            m = 1
        else:
            long_trace = h if h > w else w
            stride = 1
            stride_left = 1
            stride_right = -1
            ref = max_r * 0.6
            space = long_trace - max_r
            while stride < ref:
                while space % stride != 0:
                    stride += 1
                stride_left = stride
            while space % stride != 0:
                stride += 1
            stride_right = stride
            stride = stride_left if ref - stride_left < stride_right - ref else stride_right
            m = space // stride + 1
        # region features
        region_max = torch.zeros(size=c, device=self.__device)
        region_avg = torch.zeros(size=c, device=self.__device)
        for scale in range(1, self.__min_scale + 1):
            if max_r < scale + 1:
                break
            out_shape = (scale, scale + m - 1) if h < w else (scale + m - 1, scale)
            map_max = nn.functional.adaptive_max_pool2d(output_size=out_shape)
            map_max = torch.sum(map_max, dim=-1)
            map_max = torch.sum(map_max, dim=-1)
            l2norm_max = nn.functional.normalize(map_max)
            region_max += l2norm_max
            map_avg = nn.functional.adaptive_avg_pool2d(output_size=out_shape)
            map_avg = torch.sum(map_avg, dim=-1)
            map_avg = torch.sum(map_avg, dim=-1)
            l2norm_avg = nn.functional.normalize(map_avg)
            region_avg += l2norm_avg
        region_max = nn.functional.normalize(region_max)
        region_avg = nn.functional.normalize(region_avg)
        return torch.cat([region_avg, region_max], dim=0)

    def __score_map(self, target_feature, query_feature, upsamp_size):
        conv_map = torch.nn.functional.conv2d(target_feature, query_feature)
        data = conv_map.numpy()
        up_data = resize(data, upsamp_size)
        return torch.tensor(up_data, device=self.__device)

    def __img2tensorbatch(self, img):
        img_array = np.transpose(img, (2, 0, 1))
        img_tensor = torch.tensor(img_array / 255, dtype=torch.float, device=self.__device)
        img_tensor = torch.unsqueeze(img_tensor, 0)
        return img_tensor

    def map_fusion_features(self, img, save_path=None):
        img_tensor = self.__img2tensorbatch(img)

        img_conv_features, img_max_features = self.__feature_model.representations_of(img_tensor)
        # A&MP
        img_avg_features = [nn.functional.avg_pool2d(input=feature, kernel_size=2, stride=2)
                            for feature in img_conv_features]
        img_fusion_features = [torch.cat([img_avg_features[i], img_max_features[i]], dim=1)
                               for i in range(len(img_avg_features))]
        if save_path is not None:
            torch.save(img_fusion_features, save_path)

        return img_fusion_features

    def loc_fusion_features(self, img):
        img_tensor = self.__img2tensorbatch(img)

        img_conv_features = self.__feature_model.representations_of(img_tensor, return_max=False)
        # GA&MP
        img_fusion_features = [self.__global_am_pooling(img_conv_features[i]) for i in range(3)]
        # R-AMAC
        img_fusion_features += [self.__region_amac(img_conv_features[i]) for i in range(4, 6)]

        return img_fusion_features

    def detect_location(self, target_img, query_img, target_fusion_features=None, query_fusion_features=None,
                        save_target_feature=None, save_heat_map=None, save_region=None):
        if not target_fusion_features:
            target_fusion_features = self.map_fusion_features(target_img, save_target_feature)
        if not query_fusion_features:
            query_fusion_features = self.loc_fusion_features(query_img)
        # score maps

        score_maps = [self.__score_map(target_fusion_features[i], query_fusion_features[i],
                                       (target_img.shape[0], target_img.shape[1]))
                      for i in range(len(target_fusion_features))]
        fusion_map = torch.mean(torch.tensor(score_maps, device=self.__device), dim=0)
        # 2stage detection
        score_array = fusion_map.numpy()
        if save_heat_map is not None:
            imsave(save_heat_map, score_array)

        thred_ada = (np.mean(score_array) + np.max(score_array)) / 2
        binary_map = np.int(score_array > thred_ada)
        components = connected_components(binary_map)
        score = 0
        region = None
        for component in components:
            if len(component) > 16:
                pts = np.array(component)
                x_min, x_max, y_min, y_max = pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()
                crop = target_img[y_min:y_max + 1, x_min:x_max + 1, :].copy()
                crop_tensor = torch.tensor(crop / 255, dtype=torch.double, device=self.__device)
                crop_conv_features, crop_max_features = self.__feature_model.representations_of(crop_tensor)
                # GA&MP
                crop_fusion_features = [self.__global_am_pooling(crop_conv_features[i]) for i in range(3)]
                # R-AMAC
                crop_fusion_features += [self.__region_amac(crop_conv_features[i]) for i in range(4, 6)]
                crop_scores = [(query_fusion_features[i] * crop_fusion_features[i]).sum()
                               for i in range(len(query_fusion_features))]
                crop_score = torch.mean(torch.tensor(crop_scores, device=self.__device)).numpy()
                if crop_score > score:
                    region = (x_min, x_max, y_min, y_max)
        if save_region and region is not None:
            imsave(save_region, target_img[y_min:y_max + 1, x_min:x_max + 1, :].copy())
        return region


if __name__ == '__main__':
    # model_filename = 'vgg16_bn-6c64b313.pth'
    model_filename = 'net_checkpoint_47280.pth'
    model_file_path = os.path.join(proj_path, 'model_zoo', 'checkpoints', model_filename)
    detector = LocationDetector(model_file_path, device='cpu')
    map_village = imread(os.path.join(data_village_dir, 'map.jpg'))
    frame_files = os.listdir(os.path.join(data_village_dir, 'frames'))
    #map_features = detector.map_fusion_features(map_village, os.path.join(data_village_dir, 'map.pth'))
    map_features = torch.load(os.path.join(data_village_dir, 'map.pth'))
    for img_file in frame_files:
        save_score_map = os.path.join(expr_base, 'localization', 'score_map_' + img_file)
        save_region = os.path.join(expr_base, 'localization', 'location_' + img_file)
        img_array = imread(os.path.join(data_village_dir, 'frames', img_file))
        region = detector.detect_location(target_img=map_village, query_img=img_array,
                                          target_fusion_features=map_features,
                                          save_heat_map=save_score_map, save_region=save_region)
