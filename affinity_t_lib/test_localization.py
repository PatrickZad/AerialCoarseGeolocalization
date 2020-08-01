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
    parser = argparse.ArgumentParser(description='')

    # file/folder pathes
    parser.add_argument("--encoder_dir", type=str,
                        default='affinity_t_lib/weights/encoder_single_gpu.pth', help="pretrained encoder")
    parser.add_argument("--decoder_dir", type=str,
                        default='affinity_t_lib/weights/decoder_single_gpu.pth', help="pretrained decoder")
    parser.add_argument('--resume', type=str, default='', metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument("-c", "--savedir", type=str,
                        default="affinity_t_lib/match_track_comb/", help='checkpoints path')
    parser.add_argument("--Resnet", type=str, default="r18",
                        help="choose from r18 or r50")

    # main parameters
    parser.add_argument("--pretrainRes", action="store_true")
    parser.add_argument("--batchsize", type=int, default=1, help="batchsize")
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=256,
                        help="crop size for localization.")
    parser.add_argument("--full_size", type=int, default=640,
                        help="full size for one frame.")
    parser.add_argument("--window_len", type=int, default=2,
                        help='number of images (2 for pair and 3 for triple)')
    parser.add_argument("--temp", type=int, default=1,
                        help="temprature for softmax.")

    print("Begin parser arguments.")
    args = parser.parse_args()
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)
    args.savepatch = os.path.join(args.savedir, 'savepatch')
    args.logfile = open(os.path.join(args.savedir, "logargs.txt"), "w")
    args.multiGPU = args.device == torch.cuda.device_count()

    if not args.multiGPU:
        torch.cuda.set_device(args.device)
    if not os.path.exists(args.savepatch):
        os.mkdir(args.savepatch)

    args.vis = True
    if args.color_switch > 0:
        args.color_switch_flag = True
    else:
        args.color_switch_flag = False
    if args.coord_switch > 0:
        args.coord_switch_flag = True
    else:
        args.coord_switch_flag = False

    try:
        from tensorboardX import SummaryWriter
        global writer
        writer = SummaryWriter()
    except ImportError:
        args.vis = False
    print(' '.join(sys.argv))
    print('\n')
    args.logfile.write(' '.join(sys.argv))
    args.logfile.write('\n')

    for k, v in args.__dict__.items():
        print(k, ':', v)
        args.logfile.write('{}:{}\n'.format(k, v))
    args.logfile.close()
    return args


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
