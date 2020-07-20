import sys
import os
import loc_detect
import feat_extr
import torch.nn as nn
import torch
import skimage.io as imgio
import skimage.draw as draw
import selectivesearch
import numpy as np


class RegionSet:
    def __init__(self, h=None, w=None, t=None, b=None, l=None, r=None):
        self._reigon_set = ((0, h), (0, h), (0, w), (0, w))

    def split_region(self):
        pass


class MacLocationDetector(loc_detect.FeatureBasedLocationDetector):
    def __init__(self, extr_type=feat_extr.VGG_BN_TUNE, device='cpu' if not torch.cuda.is_available() else 'cuda',
                 stride=3, ratio_factor=1.1, optim_iter=5, optim_maxstep=3):
        assert optim_maxstep > 0 and optim_iter >= 0
        model = feat_extr.ExtractorFactory.create_feature_extractor(extr_type)
        self._down_samp = 5
        self._discrim_thred = 1
        self._stride = stride
        self._ratio_factor = ratio_factor
        self._optim_iter = 5
        self._optim_steps = list(range(-optim_maxstep, 0)) + list(range(1, optim_maxstep + 1))
        super(MacLocationDetector, self).__init__(model, device)

    def map_features(self, map_img):
        map_img_t = super(MacLocationDetector, self)._img2tensorbatch(map_img)
        _, max_block_fratures = self._extr.features_of(map_img_t)
        return max_block_fratures[-1]

    def loc_features(self, loc_img):
        loc_img_t = super(MacLocationDetector, self)._img2tensorbatch(loc_img)
        _, max_block_features = self._extr.features_of(loc_img_t)
        return max_block_features[-1]

    def _adaptive_padding(self, l_limit, l_kernel):
        padding0, padding1 = 0, 0
        if (l_limit - l_kernel) % self._stride != 0:
            num = (l_limit - l_kernel) // self._stride
            margin = (num + 1) * self._stride + l_kernel - l_limit
            padding0 = margin // 2
            padding1 = margin - padding0
        return padding0, padding1

    def _mat_argmax(self, mat_tensor, save_mat=None):
        if save_mat is not None:
            save_as_heatmap(mat_tensor.numpy(), save_mat)
        vec_tensor = torch.reshape(mat_tensor, (-1,))
        idx = torch.argmax(vec_tensor)
        h, w = mat_tensor.shape
        return torch.max(vec_tensor), torch.tensor((idx // w, idx % w))

    def _inv_conv_coord(self, padding, position):
        h, w = position
        h_m, w_m = h * self._stride, w * self._stride
        return torch.tensor((max(0, h_m - padding[0]), max(0, w_m - padding[1])))

    def _aml_search(self, map_feature, loc_mac, save_region=None, use_avg=False):
        aspect_ratio_wh = loc_mac.shape[-2] / loc_mac.shape[-1]
        aspect_ratio_hw = loc_mac.shape[-1] / loc_mac.shape[-2]
        region_limit_h, region_limit_w = map_feature.shape[-2], map_feature.shape[-1]
        best_region = None
        best_score = 0
        loc_mac_length = torch.norm(loc_mac, p=None, dim=1)
        score_region_dict = {}
        loc_mac_conv_weight = loc_mac.reshape((1, -1, 1, 1))
        for r_h in range(region_limit_h - 1, 0, -1):
            for r_w in range(region_limit_w - 1, 0, -1):
                if r_h / r_w > self._ratio_factor * aspect_ratio_hw or r_w / r_h > self._ratio_factor * aspect_ratio_wh:
                    continue
                padding_left, padding_right = self._adaptive_padding(region_limit_w, r_w)
                padding_top, padding_bottom = self._adaptive_padding(region_limit_h, r_h)
                padding_h, padding_w = max(padding_top, padding_bottom), max(padding_left, padding_right)
                if padding_h > r_h / 2 or padding_w > r_w / 2:
                    padding_h, padding_w = 0, 0
                region_mac_map = nn.functional.max_pool2d(map_feature, kernel_size=(r_h, r_w), stride=self._stride,
                                                          padding=(padding_h, padding_w))
                if use_avg:
                    avg = nn.functional.avg_pool2d(map_feature, kernel_size=(r_h, r_w), stride=self._stride,
                                                   padding=(padding_h, padding_w))
                    region_mac_map = torch.cat([region_mac_map, avg], dim=1)
                map_mac_length = torch.norm(region_mac_map, p=None, dim=1)
                length_product = loc_mac_length * map_mac_length
                length_product = torch.unsqueeze(length_product, dim=1)
                inner_product_map = nn.functional.conv2d(region_mac_map, loc_mac_conv_weight,
                                                         stride=1)
                match_map = torch.squeeze(inner_product_map / length_product)
                if match_map.min() > self._discrim_thred * match_map.max():
                    continue
                save_mat = None
                if save_region is not None:
                    save_dir = os.path.dirname(save_region)
                    dir_name = os.path.split(save_region)[-1][:-4]
                    save_dir = os.path.join(save_dir, dir_name)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_mat = os.path.join(save_dir, 'on ' + str(r_h) + 'x' + str(r_w) + '.jpg')
                scale_best_score, best_match_position = self._mat_argmax(match_map, save_mat)
                best_match_region_corner = self._inv_conv_coord(padding=(padding_h, padding_w),
                                                                position=best_match_position)
                key_score = scale_best_score.item()
                if key_score not in score_region_dict.keys():
                    score_region_dict[key_score] = torch.cat([best_match_region_corner, torch.tensor((r_h, r_w))])
                if scale_best_score > best_score:
                    best_region = torch.cat([best_match_region_corner, torch.tensor((r_h, r_w))])
                    best_score = scale_best_score
        sorted_regions = sorted(score_region_dict.items(), key=lambda item: item[0], reverse=True)
        region_arr = best_region.numpy()
        # coord descent optimization
        current_region_arr = region_arr.copy()
        current_best_score = best_score
        print('origin ' + str(current_region_arr))
        for i in range(self._optim_iter):
            optimed = False
            for var_idx in range(region_arr.shape[0]):
                var_best_step = 0
                var_best_score = current_best_score
                for step in self._optim_steps:
                    region_param = current_region_arr.copy()
                    region_param[var_idx] += step
                    param_limit = np.array([[0, region_limit_h - 1], [0, region_limit_w - 1],
                                            [1, region_limit_h - region_param[0]],
                                            [1, region_limit_w - region_param[1]]])
                    if region_param[var_idx] < param_limit[var_idx][0] or \
                            region_param[var_idx] > param_limit[var_idx][1]:
                        continue
                    print(current_region_arr)
                    region = map_feature[..., region_param[0]:region_param[0] + region_param[2],
                             region_param[1]:region_param[1] + region_param[3]]
                    if region_param[0] < 0 or region_param[1] < 0 or region_param[0] >= region_param[0] + region_param[
                        2] or region_param[1] >= region_param[1] + region_param[3]:
                        print('touch')
                    print(region.shape)
                    print('region ' + str(region_param))
                    print('current ' + str(current_region_arr))
                    region_mac = nn.functional.adaptive_max_pool2d(region, output_size=(1, 1))
                    if use_avg:
                        avg = nn.functional.adaptive_avg_pool2d(region, (1, 1))
                        region_mac = torch.cat([region_mac, avg], dim=1)
                    region_mac_length = torch.norm(region_mac, p=None, dim=1)
                    product = (region_mac * loc_mac).sum()
                    score = product / (region_mac_length * loc_mac_length)
                    if score > var_best_score:
                        var_best_step = step
                        var_best_score = score
                if var_best_step != 0:
                    print('from ' + str(current_region_arr))
                    current_region_arr[var_idx] += var_best_step
                    print('to ' + str(current_region_arr))
                    current_best_score = var_best_score
                    optimed = True
            if not optimed:
                break
        region_np = current_region_arr * 2 ** self._down_samp
        origin_arr = region_arr * 2 ** self._down_samp
        return region_np, origin_arr

    def _region_selective_search(self):
        pass

    def _feature_selective_search(self, map_img, map_feature, loc_mac, save_region=None, use_avg=False):
        aspect_ratio_wh = loc_mac.shape[-2] / loc_mac.shape[-1]
        aspect_ratio_hw = loc_mac.shape[-1] / loc_mac.shape[-2]
        img_lbl, s_regions = selectivesearch.selective_search(map_img, scale=500, sigma=0.9,
                                                              min_size=2 ** (self._down_samp * 2))
        score_region_dict = {}
        loc_mac_length = torch.norm(loc_mac, p=None, dim=1)
        for region in s_regions:
            rect = np.array(region['rect'], dtype=np.float)
            if rect[2] / rect[3] > self._ratio_factor * aspect_ratio_wh or \
                    rect[3] / rect[2] > self._ratio_factor * aspect_ratio_hw:
                continue
            feature_rect = np.int32(np.ceil(rect / (2 ** self._down_samp)))
            feature_rect[2] = min(feature_rect[2], map_feature.shape[-1] - feature_rect[0])
            feature_rect[3] = min(feature_rect[3], map_feature.shape[-2] - feature_rect[1])
            if feature_rect[2] * feature_rect[3] < 1:
                continue
            region_feature = map_feature[..., feature_rect[1]:feature_rect[1] + feature_rect[3],
                             feature_rect[0]:feature_rect[0] + feature_rect[2]]
            region_mac = nn.functional.adaptive_max_pool2d(region_feature, output_size=(1, 1))
            if use_avg:
                avg = nn.functional.adaptive_avg_pool2d(region_feature, (1, 1))
                region_mac = torch.cat([region_mac, avg], dim=1)
            region_mac_length = torch.norm(region_mac, p=None, dim=1)
            score = (loc_mac * region_mac).sum() / (loc_mac_length * region_mac_length)
            score_scalar = score.item()
            if score_scalar not in score_region_dict.keys():
                score_region_dict[score_scalar] = np.int32(rect)
            else:
                if rect[2] * rect[3] < score_region_dict[score_scalar][2] * score_region_dict[score_scalar][3]:
                    score_region_dict[score_scalar] = np.int32(rect)
        sorted_regions = sorted(score_region_dict.items(), key=lambda item: item[0], reverse=True)
        if save_region is not None:
            save_dir = os.path.dirname(save_region)
            dir_name = os.path.split(save_region)[-1][:-4]
            save_dir = os.path.join(save_dir, dir_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for i in range(min(64, len(sorted_regions))):
                score, r = sorted_regions[i]
                save_file = os.path.join(save_dir, 'r' + str(i + 1) + '_{:.6f}' '.jpg'.format(score))
                img_mat = map_img[r[1]:r[1] + r[3], r[0]:r[0] + r[2], :]
                imgio.imsave(os.path.join(save_dir, save_file), img_mat)
        best_region = sorted_regions[0][1]
        return np.int32((best_region[1], best_region[0], best_region[3], best_region[2]))

    def detect_location(self, map_img, loc_img, map_features=None, loc_features=None, save_region=None,
                        selective_search=False, use_avg=True):
        if map_features is None:
            map_features = self.map_features(map_img)
        if loc_features is None:
            loc_features = self.loc_features(loc_img)
            loc_mac_feature = nn.functional.adaptive_max_pool2d(loc_features, output_size=(1, 1))
            if use_avg:
                avg = nn.functional.adaptive_avg_pool2d(loc_mac_feature, (1, 1))
                loc_mac_feature = torch.cat([loc_mac_feature, avg], dim=1)
        if not selective_search:
            op_region_np, region_np = self._aml_search(map_features, loc_mac_feature, save_region, use_avg)
        else:
            region_np = self._feature_selective_search(map_img, map_features, loc_mac_feature, save_region, use_avg)
            op_region_np = region_np.copy()
        if save_region is not None:
            x_coords = [op_region_np[1], op_region_np[1] + op_region_np[3], op_region_np[1] + op_region_np[3],
                        op_region_np[1]]
            y_coords = [op_region_np[0], op_region_np[0], op_region_np[0] + op_region_np[2],
                        op_region_np[0] + op_region_np[2]]
            rr, cc = draw.polygon_perimeter(y_coords, x_coords, shape=map_img.shape, clip=True)
            save_img = map_img.copy()
            draw.set_color(save_img, [rr, cc], (0, 255, 0))
            x_coords = [region_np[1], region_np[1] + region_np[3], region_np[1] + region_np[3], region_np[1]]
            y_coords = [region_np[0], region_np[0], region_np[0] + region_np[2], region_np[0] + region_np[2]]
            rr, cc = draw.polygon_perimeter(y_coords, x_coords, shape=map_img.shape, clip=True)
            draw.set_color(save_img, [rr, cc], (255, 0, 0))
            # region_img = map_img[region_np[0]:region_np[0] + region_np[2], region_np[1]:region_np[1] + region_np[3], :]
            imgio.imsave(save_region, save_img)
        return region_np, op_region_np


if __name__ == '__main__':
    from common import *
    from skimage.io import imread, imsave
    from data.dataset import RemoteDataReader, OxfordBuildingsLocalization, getVHRRemoteDataRandomCropper
    import feat_extr


    def detectiong_expr(expr_subdir):
        print(expr_subdir)
        expr_out = os.path.join(proj_path, 'experiments', 'localization', expr_subdir)
        detector = MacLocationDetector(extr_type=feat_extr.RESNET50)
        img_iter, _ = getVHRRemoteDataRandomCropper()
        total_iou_det = 0
        total_iou_op = 0
        count = 0
        for map, loc, bbox in img_iter:
            save_region = os.path.join(expr_out, 'vhr', str(count) + '.jpg')
            op_result, result = detector.detect_location(map_img=map, loc_img=loc, save_region=save_region)
            # x_min,y_min,x_max,y_max
            result_box = [result[1], result[0], result[1] + result[3], result[0] + result[2]]
            op_result_box = [op_result[1], op_result[0], op_result[1] + op_result[3], op_result[0] + op_result[2]]
            loc_box = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            i_area = (min(result_box[2], loc_box[2]) - max(result_box[0], loc_box[0])) * \
                     (min(result_box[3], loc_box[3]) - max(result_box[1], loc_box[1]))
            u_area = result[2] * result[3] + bbox[2] * bbox[3] - i_area
            iou = i_area / u_area
            total_iou_det += iou
            i_area = (min(op_result_box[2], loc_box[2]) - max(op_result_box[0], loc_box[0])) * \
                     (min(op_result_box[3], loc_box[3]) - max(op_result_box[1], loc_box[1]))
            u_area = op_result[2] * op_result[3] + bbox[2] * bbox[3] - i_area
            iou = i_area / u_area
            total_iou_op += iou
            contrast_file = os.path.join(expr_out, 'vhr', str(count) + '_contrast.jpg')
            rr, cc = draw.polygon_perimeter([loc_box[1], loc_box[1], loc_box[3], loc_box[3]],
                                            [loc_box[0], loc_box[2], loc_box[2], loc_box[0]],
                                            shape=map.shape, clip=True)
            draw.set_color(map, [rr, cc], (0, 255, 0))
            rr, cc = draw.polygon_perimeter([result_box[1], result_box[1], result_box[3], result_box[3]],
                                            [result_box[0], result_box[2], result_box[2], result_box[0]],
                                            shape=map.shape, clip=True)
            draw.set_color(map, [rr, cc], (255, 0, 0))
            rr, cc = draw.polygon_perimeter([op_result_box[1], op_result_box[1], op_result_box[3], op_result_box[3]],
                                            [op_result_box[0], op_result_box[2], op_result_box[2], op_result_box[0]],
                                            shape=map.shape, clip=True)
            draw.set_color(map, [rr, cc], (0, 0, 255))
            contrast = np.zeros((max(loc.shape[0], map.shape[0]), loc.shape[1] + map.shape[1] + 16, 3), dtype=np.uint8)
            contrast[:loc.shape[0], :loc.shape[1], :] = loc
            contrast[:map.shape[0], loc.shape[1] + 16:, :] = map
            imsave(contrast_file, contrast)

            count += 1

        print('det iou: ' + str(total_iou_det / count))
        print('op iou: ' + str(total_iou_op / count))

        map_village = imread(os.path.join(data_village_dir, 'map.jpg'))
        frame_files = os.listdir(os.path.join(data_village_dir, 'frames'))
        map_features = detector.map_features(map_village)
        for img_file in frame_files:
            save_region = os.path.join(expr_out, 'village', 'location_' + img_file)
            img_array = imread(os.path.join(data_village_dir, 'frames', img_file))
            detector.detect_location(map_img=map_village, loc_img=img_array,
                                     map_features=map_features,
                                     save_region=save_region)
        map_gravel = imread(os.path.join(data_gravel_dir, 'map.jpg'))
        frame_files = os.listdir(os.path.join(data_gravel_dir, 'frames'))
        map_features = detector.map_features(map_gravel)
        for img_file in frame_files:
            save_region = os.path.join(expr_out, 'gravel_pit', 'location_' + img_file)
            img_array = imread(os.path.join(data_gravel_dir, 'frames', img_file))
            detector.detect_location(map_img=map_gravel, loc_img=img_array,
                                     map_features=map_features,
                                     save_region=save_region)
        rs_data = RemoteDataReader()
        for id, target, query in rs_data:
            target_features = detector.map_features(target)
            save_region = os.path.join(expr_out, 'remote', 'location_' + id + '.jpg')
            detector.detect_location(map_img=target, loc_img=query,
                                     map_features=target_features,
                                     save_region=save_region)
        bd = OxfordBuildingsLocalization()
        for ids, img_ts in bd:
            target_features = detector.map_features(img_ts[1])
            save_region = os.path.join(expr_out, 'oxford', ids[0] + '_on_' + ids[1] + '.jpg')
            detector.detect_location(map_img=img_ts[1], loc_img=img_ts[0],
                                     map_features=target_features,
                                     save_region=save_region)


    detectiong_expr('mac_res50_avg')
