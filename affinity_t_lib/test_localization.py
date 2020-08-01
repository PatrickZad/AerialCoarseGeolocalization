# OS libraries
import os
import copy
import queue
import argparse
import scipy.misc
import numpy as np

# Pytorch
import torch
import torch.nn as nn

# Customized libraries
from affinity_t_lib.libs.test_utils import *
from affinity_t_lib.libs.model import transform
from affinity_t_lib.libs.utils import norm_mask
from affinity_t_lib.libs.model import track_match_comb as Model


############################## helper functions ##############################
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--out_dir", type=str, default="results/",
                        help="output saving path")

    parser.add_argument("-c", "--checkpoint_dir", type=str,
                        default="weights/checkpoint_latest.pth.tar",
                        help="checkpoints path")
    args = parser.parse_args()
    args.is_train = False
    return args


############################## testing functions ##############################

def forward(frame1, frame2, model):
    n, c, h1, w1 = frame1.size()
    n, c, h2, w2 = frame2.size()
    frame1_gray = frame1[:, 0].view(n, 1, h1, w1)
    frame2_gray = frame2[:, 0].view(n, 1, h2, w2)
    frame1_gray = frame1_gray.repeat(1, 3, 1, 1)
    frame2_gray = frame2_gray.repeat(1, 3, 1, 1)

    output = model(frame1_gray, frame2_gray, frame1, frame2)
    # top left and bottom right
    bbox = output[2]

    return bbox.numpy()


############################## main function ##############################

if (__name__ == '__main__'):
    from data.dataset import SenseflyTransVal
    from torch.utils.data import DataLoader
    import cv2

    args = parse_args()

    # loading pretrained model
    model = Model(pretrained=False, )
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(args.checkpoint_dir)
    best_loss = checkpoint['best_loss']
    model.load_state_dict(checkpoint['state_dict'])
    model = model.module
    print("=> loaded checkpoint '{} ({})' (epoch {})"
          .format(args.checkpoint_dir, best_loss, checkpoint['epoch']))
    model.cuda()
    model.eval()

    # start testing
    scale_f = 0.6
    dataset = SenseflyTransVal(scale_f=scale_f)
    loader = DataLoader(dataset, batch_size=1)
    dataset_dir = dataset.get_dataset_dir()
    save_dir = args.out_dir
    for env_dir, img_t, img_file, map_t, map_file in loader:
        img_t = img_t.cuda()
        map_t = map_t.cuda()

        img_arr = cv2.imread(os.path.join(dataset_dir, env_dir[0], 'imgs', img_file[0]))
        img_arr = cv2.resize(img_arr, (0, 0), fx=scale_f, fy=scale_f)
        map_arr = cv2.imread(os.path.join(dataset_dir, env_dir[0], 'map', map_file[0]))
        map_arr = cv2.resize(map_arr, (0, 0), fx=scale_f, fy=scale_f)
        loc_box = model(img_t, map_t, False, False, patch_size=[img_arr.shape[0] // 8, img_arr.shape[1] // 8])
        pts = np.array(
            [[loc_box[0], loc_box[1]], [loc_box[2], loc_box[1]], [loc_box[2], loc_box[3]], [loc_box[0], loc_box[3]]],
            np.int32)
        cv2.polylines(map_arr, pts, True, (0, 0, 255), thickness=5)
        background = np.zeros((max(img_arr.shape[0], map_arr.shape[0]), img_arr.shape[0] + map_arr.shape[0], 3))
        background[:img_arr.shape[0], :img_arr.shape[1], :] = img_arr
        background[:map_arr.shape[0], img_arr.shape[1]:img_arr.shape[1] + map_arr.shape[1], :] = map_arr
        cv2.imwrite(os.path.join(save_dir, env_dir + '_' + img_file))

    '''
    for cnt, line in enumerate(lines):
        video_nm = line.strip()
        print('[{:n}/{:n}] Begin to segmentate video {}.'.format(cnt, len(lines), video_nm))

        video_dir = os.path.join(args.davis_dir, video_nm)
        frame_list = read_frame_list(video_dir)
        seg_dir = frame_list[0].replace("JPEGImages", "Annotations")
        seg_dir = seg_dir.replace("jpg", "png")
        _, first_seg, seg_ori = read_seg(seg_dir, args.scale_size)
        test(model, frame_list, video_dir, first_seg, seg_ori)
    '''
