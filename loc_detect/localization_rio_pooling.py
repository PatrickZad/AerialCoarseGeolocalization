import loc_detect
import feat_extr
import torch.nn as nn
import torch
import skimage.io as imgio
from common import save_as_heatmap
import os


class RoipoolingDetector(loc_detect.FeatureBasedLocationDetector):
    def __init__(self, extr_type=feat_extr.VGG_TUNE, max_roi_scale=4,
                 device='cpu' if not torch.cuda.is_available() else 'cuda', ):
        model = feat_extr.ExtractorFactory.create_feature_extractor(extr_type)
        self._down_samp = 5
        self._max_roi_scale = max_roi_scale
        super(RoipoolingDetector, self).__init__(model, device)

    def map_features(self, map_img):
        map_img_t = super(RoipoolingDetector, self)._img2tensorbatch(map_img)
        _, max_block_fratures = self._extr.features_of(map_img_t)
        return max_block_fratures[-1]

    def loc_features(self, loc_img):
        loc_img_t = super(RoipoolingDetector, self)._img2tensorbatch(loc_img)
        _, max_block_features = self._extr.features_of(loc_img_t)
        return max_block_features[-1]

    def _save_scale_heatmap(self, conv_result, save_dir, map_size):
        score_maps = (torch.squeeze(conv_result, dim=0)).numpy()
        chs = score_maps.shape[0]
        for c in range(chs):
            file_name = str(c) + '.jpg'
            save_as_heatmap(score_maps[c, ...], os.path.join(save_dir, file_name))
        mean_map = np.mean(score_maps, axis=0)
        score_map = cv.resize(mean_map, dsize=(map_size[1], map_size[0]), interpolation=cv.INTER_CUBIC)
        save_as_heatmap(score_map, os.path.join(save_dir, 'mean.jpg'))

    def detect_location(self, map_img, loc_img, map_features=None, loc_features=None, save_region=None):
        if map_features is None:
            map_features = self.map_features(map_img)
        if loc_features is None:
            loc_features = self.loc_features(loc_img)
        loc_h, loc_w = loc_features.shape[-2], loc_features.shape[-1]
        d = min(loc_h, loc_w)
        for out_d in range(self._max_roi_scale, d):
            pooling_size = (out_d, int(out_d / d * loc_w)) if loc_h < loc_w else (int(out_d / d * loc_h), out_d)
            roi_pooling_map = nn.functional.adaptive_max_pool2d(loc_features, output_size=pooling_size)
            bs, ch, mh, mw = roi_pooling_map.shape
            kernel_weights = []
            for wi in range(mw):
                for hi in range(mh):
                    kernel_weights.append(roi_pooling_map[:, :, hi:hi + 1, wi:wi + 1])
            conv_weight = torch.cat(kernel_weights, dim=0)
            '''conv_weight = torch.tensor([roi_pooling_map[:, :, hi, wi] for wi in range(mw) for hi in range(mh)],
                                       device=self._device)'''
            scale_det_map = nn.functional.conv2d(map_features, conv_weight)
            if save_region is not None:
                scale_dir = str(pooling_size[0]) + 'x' + str(pooling_size[1])
                save_dir = os.path.dirname(save_region)
                dir_name = os.path.split(save_region)[-1][:-4]
                save_dir = os.path.join(save_dir, dir_name, scale_dir)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                self._save_scale_heatmap(scale_det_map, save_dir, map_img.shape[:-1])


if __name__ == '__main__':
    from common import *
    from skimage.io import imread
    from data.dataset import RemoteDataReader
    import feat_extr


    def detectiong_expr(expr_subdir):
        print(expr_subdir)
        expr_out = os.path.join(proj_path, 'experiments', 'localization', expr_subdir)
        detector = RoipoolingDetector(extr_type=feat_extr.VGG_TUNE)

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


    detectiong_expr('roi_vgg_tune')
