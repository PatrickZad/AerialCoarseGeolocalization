# OS libraries
import os
import argparse
import sys

# Pytorch
import torch
import torch.nn as nn

# Customized libraries
from affinity_t_lib.libs.test_utils import *
from affinity_t_lib.libs.model import transform
from affinity_t_lib.libs.utils import norm_mask

from affinity_t_lib.model import track_match_comb as Model


def parse_args():
    parser = argparse.ArgumentParser(description='')

    # file/folder pathes

    parser.add_argument("--encoder_dir", type=str, default='affinity_t_lib/weights/encoder_single_gpu.pth',
                        help="pretrained encoder")
    parser.add_argument("--decoder_dir", type=str, default='affinity_t_lib/weights/decoder_single_gpu.pth',
                        help="pretrained decoder")
    parser.add_argument('--resume', type=str, default='affinity_t_lib/weights/checkpoint_latest.pth.tar', metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument("-c", "--savedir", type=str, default="match_track_comb/", help='checkpoints path')
    parser.add_argument("--Resnet", type=str, default="r18", help="choose from r18 or r50")

    # main parameters
    parser.add_argument("--pretrainRes", action="store_true")
    parser.add_argument("--batchsize", type=int, default=1, help="batchsize")
    parser.add_argument('--workers', type=int, default=16)

    parser.add_argument("--patch_size", type=int, default=256, help="crop size for localization.")
    parser.add_argument("--full_size", type=int, default=640, help="full size for one frame.")
    parser.add_argument("--window_len", type=int, default=2, help='number of images (2 for pair and 3 for triple)')
    parser.add_argument("--device", type=int, default=4,
                        help="0~device_count-1 for single GPU, device_count for dataparallel.")
    parser.add_argument("--temp", type=int, default=1, help="temprature for softmax.")

    print("Begin parser arguments.")
    args = parser.parse_args()
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)
    args.savepatch = os.path.join(args.savedir, 'savepatch')
    if not os.path.exists(args.savepatch):
        os.mkdir(args.savepatch)
    return args


if (__name__ == '__main__'):
    from data.dataset import SenseflyTransVal
    from torch.utils.data import DataLoader
    import cv2

    args = parse_args()
    # loading pretrained model
    model = Model(args.pretrainRes, args.encoder_dir, args.decoder_dir, temp=args.temp, Resnet=args.Resnet,
                  color_switch=False, coord_switch=False)
    #model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(args.resume)
    best_loss = checkpoint['best_loss']
    model.load_state_dict(checkpoint['state_dict'])
    #model = model.module
    model.cuda()
    model.eval()

    # start testing
    scale_f = 0.4
    dataset = SenseflyTransVal(scale_f=scale_f)
    loader = DataLoader(dataset, batch_size=1)
    dataset_dir = dataset.get_dataset_dir()
    save_dir = args.savedir
    for env_dir, img_t, img_file, map_t, map_file in loader:
        img_t = img_t.cuda()
        map_t = map_t.cuda()

        img_arr = cv2.imread(os.path.join(dataset_dir, env_dir[0], 'imgs', img_file[0]))
        img_arr = cv2.resize(img_arr, (0, 0), fx=scale_f, fy=scale_f)
        map_arr = cv2.imread(os.path.join(dataset_dir, env_dir[0], 'map', map_file[0]))
        map_arr = cv2.resize(map_arr, (0, 0), fx=scale_f, fy=scale_f)
        loc_box = model(img_t, map_t, False, patch_size=[img_arr.shape[0] // 8, img_arr.shape[1] // 8], nc_only=True)
        pts = np.array(
            [[[loc_box[0], loc_box[1]], [loc_box[2], loc_box[1]], [loc_box[2], loc_box[3]], [loc_box[0], loc_box[3]]]],
            np.int32)
        cv2.polylines(map_arr, pts, True, (0, 0, 255), thickness=5)
        background = np.zeros((max(img_arr.shape[0], map_arr.shape[0]), img_arr.shape[1] + map_arr.shape[1], 3))
        background[:img_arr.shape[0], :img_arr.shape[1], :] = img_arr
        background[:map_arr.shape[0], img_arr.shape[1]:img_arr.shape[1] + map_arr.shape[1], :] = map_arr
        cv2.imwrite(os.path.join(save_dir, env_dir[0] + '_' + img_file[0][:-4]+map_file[0]),background)
