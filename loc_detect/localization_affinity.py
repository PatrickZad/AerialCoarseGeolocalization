import loc_detect
import torch
import feat_extr
import torch.nn as nn
import numpy as np
import skimage.io as io


def _coord_of_idx(idx, w):
    return idx % w, idx // w


class AffinityDetector(loc_detect.FeatureBasedLocationDetector):
    def __init__(self, extr_type=feat_extr.RESNET50, device='cpu' if not torch.cuda.is_available() else 'cuda'):
        model = feat_extr.ExtractorFactory.create_feature_extractor(extr_type)
        super(AffinityDetector, self).__init__(model, device)

    def map_features(self, map_img):
        map_img_t = super(AffinityDetector, self)._img2tensorbatch(map_img)
        _, max_block_fratures = self._extr.features_of(map_img_t)
        return max_block_fratures[-1]

    def loc_features(self, loc_img):
        loc_img_t = super(AffinityDetector, self)._img2tensorbatch(loc_img)
        _, max_block_features = self._extr.features_of(loc_img_t)
        return max_block_features[-1]

    def detect_location(self, map_img, loc_img, map_feature=None, loc_feature=None, save_region=None):
        if map_feature is None:
            map_feature = self.map_features(map_img)
        if loc_feature is None:
            loc_feature = self.loc_features(loc_img)
        h_map, w_map = map_feature.shape[-2:]
        h_loc, w_loc = loc_feature.shape[-2:]
        map_coords = [(i, j) for i in range(w_map) for j in range(h_map)]
        map_feature_horizon = torch.reshape(map_feature, (1, -1, 1, h_map * w_map))
        loc_feature_vertical = torch.reshape(loc_feature, (1, -1, h_loc * w_loc, 1))
        map_feature_norm = torch.norm(map_feature_horizon, p=None, dim=1)
        loc_feature_norm = torch.norm(loc_feature_vertical, p=None, dim=1)
        map_feature_horizon = map_feature_horizon / map_feature_norm
        loc_feature_vertical = loc_feature_vertical / loc_feature_norm
        product = loc_feature_vertical * map_feature_horizon
        affinity = torch.sum(product, dim=1)
        affinity = nn.functional.softmax(affinity, dim=-1)
        map_coords = torch.tensor(map_coords, device=self._device, dtype=torch.float)
        map_coords = torch.transpose(map_coords, -1, -2)
        map_coords = torch.reshape(map_coords, shape=(1, 2, 1, -1))
        estimate_coords = torch.sum(map_coords * affinity, -1)
        estimate_coords = torch.squeeze(estimate_coords)
        estimate_coords = torch.transpose(estimate_coords, -1, -2)
        np_coords = estimate_coords.numpy()
        region_coords = np_coords * self._extr._downsamp
        if save_region is not None:
            draw_coords = np.int32(region_coords)
            length = draw_coords.shape[0]
            save_img = map_img.copy()
            for i in range(length):
                coord = draw_coords[i, :]
                left = max(coord[0] - 1, 0)
                right = min(coord[0] + 1, map_img.shape[1] - 1)
                top = max(coord[1] - 1, 0)
                bottom = min(coord[1] + 1, map_img.shape[0] - 1)
                save_img[top:bottom + 1, left:right + 1, 0] = 255
                save_img[top:bottom + 1, left:right + 1, 1:] = 0
            save_dir = os.path.dirname(save_region)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            io.imsave(save_region, save_img)
        # left,right,top,bottom
        bounding = (region_coords[:, 0].min(), region_coords[:, 0].max(),
                    region_coords[:, 1].min(), region_coords[:, 1].max(),)
        return bounding


if __name__ == '__main__':
    from common import *
    from skimage.io import imread
    from data.dataset import RemoteDataReader
    import feat_extr


    def detectiong_expr(expr_subdir):
        print(expr_subdir)
        expr_out = os.path.join(proj_path, 'experiments', 'localization', expr_subdir)
        detector = AffinityDetector()

        map_village = imread(os.path.join(data_village_dir, 'map.jpg'))
        frame_files = os.listdir(os.path.join(data_village_dir, 'frames'))
        map_features = detector.map_features(map_village)
        for img_file in frame_files:
            save_region = os.path.join(expr_out, 'village', 'location_' + img_file)
            img_array = imread(os.path.join(data_village_dir, 'frames', img_file))
            detector.detect_location(map_img=map_village, loc_img=img_array,
                                     map_feature=map_features,
                                     save_region=save_region)
        map_gravel = imread(os.path.join(data_gravel_dir, 'map.jpg'))
        frame_files = os.listdir(os.path.join(data_gravel_dir, 'frames'))
        map_features = detector.map_features(map_gravel)
        for img_file in frame_files:
            save_region = os.path.join(expr_out, 'gravel_pit', 'location_' + img_file)
            img_array = imread(os.path.join(data_gravel_dir, 'frames', img_file))
            detector.detect_location(map_img=map_gravel, loc_img=img_array,
                                     map_feature=map_features,
                                     save_region=save_region)
        rs_data = RemoteDataReader()
        for id, target, query in rs_data:
            target_features = detector.map_features(target)
            save_region = os.path.join(expr_out, 'remote', 'location_' + id + '.jpg')
            detector.detect_location(map_img=target, loc_img=query,
                                     map_feature=target_features,
                                     save_region=save_region)


    detectiong_expr('afffinity_res50')
