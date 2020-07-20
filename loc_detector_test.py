import loc_detect
import logging
import os
from common import *
from skimage.io import imread
from data.dataset import RemoteDataReader


def detectiong_expr(expr_subdir, detector):
    expr_out = os.path.join(proj_path, 'experiments', 'localization', expr_subdir)
    logging.basicConfig(filename=os.path.join(expr_out, 'test_log'),
                        level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('test_logger')
    map_village = imread(os.path.join(data_village_dir, 'map.jpg'))
    frame_files = os.listdir(os.path.join(data_village_dir, 'frames'))
    map_features = detector.map_features(map_village)
    for img_file in frame_files:
        save_region = os.path.join(expr_base, 'localization', 'village', 'location_' + img_file)
        img_array = imread(os.path.join(data_village_dir, 'frames', img_file))
        detector.detect_location(target_img=map_village, query_img=img_array,
                                 target_fusion_features=map_features,
                                 save_heat_map=save_score_map, save_region=save_region)
    map_gravel = imread(os.path.join(data_gravel_dir, 'map.jpg'))
    frame_files = os.listdir(os.path.join(data_gravel_dir, 'frames'))
    map_features = detector.map_features(map_gravel)
    for img_file in frame_files:
        save_score_map = os.path.join(expr_base, 'localization', 'gravel_pit', expr_subdir,
                                      'score_map_' + img_file)
        save_region = os.path.join(expr_base, 'localization', 'gravel_pit', 'location_' + img_file)
        img_array = imread(os.path.join(data_gravel_dir, 'frames', img_file))
        detector.detect_location(target_img=map_gravel, query_img=img_array,
                                 target_fusion_features=map_features,
                                 save_heat_map=save_score_map, save_region=save_region)
    rs_data = RemoteDataReader()
    for id, target, query in rs_data:
        target_features = detector.map_features(target)
        save_score_map = os.path.join(expr_base, 'localization', 'remote', expr_subdir, 'score_map_' + id + '.jpg')
        detector.detect_location(target_img=target, query_img=query,
                                 target_fusion_features=target_features,
                                 save_heat_map=save_score_map)
