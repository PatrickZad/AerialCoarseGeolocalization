from backbone.model import VGG16FeatureExtractor
from common import *
import os
from skimage.transform import resize
from skimage.io import imsave, imread

from backbone.model import VGG16FeatureExtractor
import torch
import torch.nn as nn

from data.dataset import RemoteDataReader
import numpy as np
import cv2 as cv


class LocationDetector:
    def __init__(self, feature_model=None, bn=True, device='cuda' if torch.cuda.is_available() else 'cpu',
                 min_scale=512, repeat_scale=True):

        self._feature_model = VGG16FeatureExtractor(vgg16_file=feature_model, device=device, bn=bn) if feature_model \
            else VGG16FeatureExtractor(device=device, bn=bn)
        self._device = device
        self._min_scale = min_scale
        self._repeat_scale = repeat_scale

    def _global_am_pooling(self, feature_map):
        '''max_map = feature_map.max(dim=-1)[0]
        max_vector = max_map.max(dim=-1)[0]
        sum_map = torch.sum(feature_map, dim=-1)
        sum_map = torch.sum(sum_map, dim=-1)
        avg_vector = sum_map / (feature_map.shape[-1] * feature_map.shape[-2])'''
        max_vector = torch.nn.functional.adaptive_max_pool2d(feature_map, output_size=(1, 1))
        avg_vector = torch.nn.functional.adaptive_avg_pool2d(feature_map, output_size=(1, 1))
        '''channel_size = feature_map.shape[0]
        max_vector = torch.tensor([feature_map[i, :, :].max() 
                                   for i in range(channel_size)], device=self.__device)
        avg_vector = torch.tensor([feature_map[i, :, :].mean() 
                                   for i in range(channel_size)], device=self.__device)'''
        return torch.cat([avg_vector, max_vector], dim=-1)

    def _region_amac(self, feature_map):
        c, h, w = feature_map.shape[1:]  # feature_map.shape[1], feature_map.shape[2]
        short_side, long_side = (h, w) if h < w else (w, h)
        if h == w:
            m = 1
            stride = int(h * 0.6) + 1
        else:
            long_trace = h if h > w else w
            stride = 1
            stride_left = 1
            stride_right = -1
            ref = short_side * 0.6
            space = long_trace - short_side
            while stride < ref and stride <= space:
                while space % stride != 0:
                    stride += 1
                stride_left = stride
                stride += 1
            if stride > space:
                stride = space
            while space % stride != 0 and stride < space:
                stride += 1
            stride_right = stride
            stride = stride_left if ref - stride_left < stride_right - ref else stride_right
            m = space // stride + 1
        # region features
        region_max = torch.zeros(size=(1, c), device=self._device)
        region_avg = torch.zeros(size=(1, c), device=self._device)
        # scale=1
        map_max = nn.functional.max_pool2d(feature_map, kernel_size=short_side, stride=stride)
        map_avg = nn.functional.avg_pool2d(feature_map, kernel_size=short_side, stride=stride)

        map_max = nn.functional.normalize(map_max, dim=1)
        map_max = torch.sum(map_max, dim=-1)
        map_max = torch.sum(map_max, dim=-1)
        # map_max = nn.functional.normalize(map_max)

        map_avg = nn.functional.normalize(map_avg, dim=1)
        map_avg = torch.sum(map_avg, dim=-1)
        map_avg = torch.sum(map_avg, dim=-1)
        # map_avg = nn.functional.normalize(map_avg)
        region_max += map_max
        region_avg += map_avg
        last_region_w = short_side
        for scale in range(2, self._min_scale + 1):
            '''if short_side < scale + 1:
                break'''
            if 2 * short_side / (scale + 1) < 1:
                break
            region_w = 2 * short_side // (scale + 1)
            if not self._repeat_scale:
                if region_w == last_region_w:
                    continue
            last_region_w = region_w
            stride_s = short_side // (scale + 1)
            stride_l = (long_side - region_w) // (scale + m - 2)
            if stride_s == 0:
                stride_s = 1
            if stride_l == 0:
                stride_l = 1
            pooling_stride = (stride_s, stride_l) if h < w else (stride_l, stride_s)
            map_max = nn.functional.max_pool2d(feature_map,
                                               kernel_size=region_w, stride=pooling_stride)
            map_avg = nn.functional.avg_pool2d(feature_map,
                                               kernel_size=region_w, stride=pooling_stride)

            map_max = nn.functional.normalize(map_max, dim=1)
            map_max = torch.sum(map_max, dim=-1)
            map_max = torch.sum(map_max, dim=-1)
            # map_max = nn.functional.normalize(map_max)

            map_avg = nn.functional.normalize(map_avg, dim=1)
            map_avg = torch.sum(map_avg, dim=-1)
            map_avg = torch.sum(map_avg, dim=-1)
            # map_avg = nn.functional.normalize(map_avg)
            region_max += map_max
            region_avg += map_avg
            '''out_shape = (scale, scale + m - 1) if h < w else (scale + m - 1, scale)
            map_max = nn.functional.adaptive_max_pool2d(feature_map, output_size=out_shape)
            map_max = torch.sum(map_max, dim=-1)
            map_max = torch.sum(map_max, dim=-1)
            l2norm_max = nn.functional.normalize(map_max)
            region_max += l2norm_max
            map_avg = nn.functional.adaptive_avg_pool2d(feature_map, output_size=out_shape)
            map_avg = torch.sum(map_avg, dim=-1)
            map_avg = torch.sum(map_avg, dim=-1)
            l2norm_avg = nn.functional.normalize(map_avg)
            region_avg += l2norm_avg'''
        region_max = nn.functional.normalize(region_max)
        region_avg = nn.functional.normalize(region_avg)
        return torch.cat([region_avg, region_max], dim=-1)

    def _score_map(self, target_feature, query_feature, upsamp_size):
        weight = query_feature.reshape((1, -1, 1, 1))
        conv_map = torch.nn.functional.conv2d(target_feature, weight)
        conv_map = conv_map.squeeze()
        data = conv_map.numpy()
        up_data = resize(data, upsamp_size)
        return up_data
        # return torch.tensor(up_data, device=self.__device)

    def _img2tensorbatch(self, img):
        img_array = np.transpose(img, (2, 0, 1))
        img_tensor = torch.tensor(img_array / 255, dtype=torch.float, device=self._device)
        img_tensor = torch.unsqueeze(img_tensor, 0)
        return img_tensor

    def map_fusion_features(self, img, save_path=None):
        img_tensor = self._img2tensorbatch(img)

        img_conv_features, img_max_features = self._feature_model.representations_of(img_tensor)
        # A&MP
        img_avg_features = [nn.functional.avg_pool2d(input=feature, kernel_size=2, stride=2)
                            for feature in img_conv_features]
        img_fusion_features = [torch.cat([img_avg_features[i], img_max_features[i]], dim=1)
                               for i in range(len(img_avg_features))]
        '''# A&MP
        img_avg_features = [nn.functional.avg_pool2d(input=feature, kernel_size=2, stride=2)
                            for feature in img_max_features]
        # img_avg_features = [nn.functional.normalize(input=feature, dim=1)
        #                     for feature in img_avg_features]
        img_max2_features = [nn.functional.max_pool2d(input=feature, kernel_size=2, stride=2)
                             for feature in img_max_features]
        # img_max2_features = [nn.functional.normalize(input=feature, dim=1)
        #                      for feature in img_max2_features]
        img_fusion_features = [torch.cat([img_avg_features[i], img_max2_features[i]], dim=1)
                               for i in range(len(img_avg_features))]'''
        if save_path is not None:
            torch.save(img_fusion_features, save_path)

        return img_fusion_features

    def loc_fusion_features(self, img):
        img_tensor = self._img2tensorbatch(img)

        # img_conv_features = self.__feature_model.representations_of(img_tensor, return_max=False)
        img_conv_features = self._feature_model.representations_of(img_tensor, return_max=False)
        # GA&MP
        img_fusion_features = [self._global_am_pooling(img_conv_features[i]) for i in range(3)]
        # R-AMAC
        img_fusion_features += [self._region_amac(img_conv_features[i]) for i in range(3, 5)]

        return img_fusion_features

    def detect_location(self, target_img, query_img, target_fusion_features=None, query_fusion_features=None,
                        save_target_feature=None, save_heat_map=None, save_region=None):
        if not target_fusion_features:
            target_fusion_features = self.map_fusion_features(target_img, save_target_feature)
        if not query_fusion_features:
            query_fusion_features = self.loc_fusion_features(query_img)
        # score maps

        score_maps = [self._score_map(target_fusion_features[i], query_fusion_features[i],
                                      (target_img.shape[0], target_img.shape[1]))
                      for i in range(len(target_fusion_features))]
        # fusion_map = torch.mean(torch.tensor(score_maps, device=self.__device), dim=0)
        fusion_map = np.array(score_maps)
        # 2stage detection
        score_array = fusion_map.mean(axis=0)
        if save_heat_map is not None:
            save_dir, file_name = os.path.split(save_heat_map)
            file_id = file_name[:-4]
            heat_dir = os.path.join(save_dir, file_id, 'heat_maps')
            gray_dir = os.path.join(save_dir, file_id, 'gray_maps')
            file_format = save_heat_map[-4:]
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if not os.path.exists(heat_dir):
                os.makedirs(heat_dir)
            if not os.path.exists(gray_dir):
                os.makedirs(gray_dir)

            '''
            
            for i in range(len(score_maps)):
                
            '''
            float_01_img = score_array / score_array.max()
            int8_img = np.uint8(float_01_img * 255)
            heat_map = cv.applyColorMap(int8_img, cv.COLORMAP_JET)
            cv.imwrite(save_heat_map, heat_map)
            imsave(save_heat_map[:-4] + '_gray' + save_heat_map[-4:], score_array)
            for i in range(len(score_maps)):
                float_img = score_maps[i] / score_maps[i].max()
                int_img = np.uint8(float_img * 255)
                sub_heat_map = cv.applyColorMap(int_img, cv.COLORMAP_JET)
                cv.imwrite(os.path.join(heat_dir, file_id + '_s' + str(i) + file_format), sub_heat_map)
                imsave(os.path.join(gray_dir, file_id + '_s' + str(i) + file_format), score_maps[i])
            ''''''
        '''thred_ada = (np.mean(score_array) + np.max(score_array)) / 2
        binary_map = np.int(score_array > thred_ada)
        components = connected_components(binary_map)
        score = 0'''
        region = None
        '''for component in components:
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
            imsave(save_region, target_img[y_min:y_max + 1, x_min:x_max + 1, :].copy())'''
        return region


if __name__ == '__main__':
    expr_config_base = {'expr_subdir': 'base',
                        'model_filename': 'net_nobn_ep100.pth',
                        'fusion_min_scale': 512,
                        'bn': False,
                        'fusion_scale_repeat': True}
    expr_config_minscale = {'expr_subdir': 'scale_m8',
                            'model_filename': 'net_nobn_ep100.pth',
                            'fusion_min_scale': 8,
                            'bn': False,
                            'fusion_scale_repeat': True}
    expr_config_repeat_scale = {'expr_subdir': 'no_repeat_fusion',
                                'model_filename': 'net_nobn_ep100.pth',
                                'fusion_min_scale': 512,
                                'bn': False,
                                'fusion_scale_repeat': False}
    expr_config_bn = {'expr_subdir': 'with_bn',
                      'model_filename': 'net_bn_final.pth',
                      'fusion_min_scale': 512,
                      'bn': False,
                      'fusion_scale_repeat': True}
    expr_config_vgg = {'expr_subdir': 'vgg',
                       'model_filename': None,
                       'fusion_min_scale': 512,
                       'bn': False,
                       'fusion_scale_repeat': True}
    expr_configs = [expr_config_bn, expr_config_vgg]


    def detection_expr(*, expr_subdir, model_filename=None, fusion_min_scale=512, bn=False, fusion_scale_repeat=True):
        print(expr_subdir)
        expr_sub_dir = expr_subdir
        model_filename = model_filename
        model_file_path = os.path.join(proj_path, 'model_zoo', 'checkpoints',
                                       model_filename) if model_filename is not None else None
        detector = LocationDetector(model_file_path, device='cpu', bn=bn, min_scale=fusion_min_scale,
                                    repeat_scale=fusion_scale_repeat)
        map_village = imread(os.path.join(data_village_dir, 'map.jpg'))
        frame_files = os.listdir(os.path.join(data_village_dir, 'frames'))
        map_features = detector.map_fusion_features(map_village)
        for img_file in frame_files:
            save_score_map = os.path.join(expr_base, 'localization', 'village', expr_sub_dir, 'score_map_' + img_file)
            save_region = os.path.join(expr_base, 'localization', 'village', 'location_' + img_file)
            img_array = imread(os.path.join(data_village_dir, 'frames', img_file))
            detector.detect_location(target_img=map_village, query_img=img_array,
                                     target_fusion_features=map_features,
                                     save_heat_map=save_score_map, save_region=save_region)
        map_gravel = imread(os.path.join(data_gravel_dir, 'map.jpg'))
        frame_files = os.listdir(os.path.join(data_gravel_dir, 'frames'))
        map_features = detector.map_fusion_features(map_gravel)
        for img_file in frame_files:
            save_score_map = os.path.join(expr_base, 'localization', 'gravel_pit', expr_sub_dir,
                                          'score_map_' + img_file)
            save_region = os.path.join(expr_base, 'localization', 'gravel_pit', 'location_' + img_file)
            img_array = imread(os.path.join(data_gravel_dir, 'frames', img_file))
            detector.detect_location(target_img=map_gravel, query_img=img_array,
                                     target_fusion_features=map_features,
                                     save_heat_map=save_score_map, save_region=save_region)
        rs_data = RemoteDataReader()
        for id, target, query in rs_data:
            target_features = detector.map_fusion_features(target)
            save_score_map = os.path.join(expr_base, 'localization', 'remote', expr_sub_dir, 'score_map_' + id + '.jpg')
            detector.detect_location(target_img=target, query_img=query,
                                     target_fusion_features=target_features,
                                     save_heat_map=save_score_map)


    for cfg in expr_configs:
        detection_expr(**cfg)
