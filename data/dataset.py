import os
import torch
from torch.utils.data import Dataset
from common import *
from skimage import io, transform
import numpy as np
import data.augmentation as aug

global iter
iter = 0


class ResiscDataset(Dataset):
    # 128*128 patches
    def __init__(self, file_path, device, categories):
        self.common_path = os.path.join(dataset_common_dir, 'NWPU-RESISC45')
        self.file_paths = file_path
        self.device = device
        self.categories = categories
        self.img_widths = range(224, 385)

    def __getitem__(self, idx):
        global iter
        img_array = io.imread(os.path.join(self.common_path, self.file_paths[idx]))  # h * w *c RGB image array
        img_class = self.file_paths[idx].split('/')[0]
        class_idx = self.categories[img_class]
        '''# random rotation
        rot_img, corners = aug.adaptive_rot(img_array)
        # square patch
        crop = aug.center_square(rot_img, corners)'''
        # random scale
        idx = np.random.randint(0, len(self.img_widths))
        img_width = self.img_widths[idx]
        # random homography
        scaled_img = transform.resize(img_array, (img_width, img_width))
        x_offset, y_offset = 0, 0
        if img_width > 224:
            x_offset = np.random.randint(0, img_width - 224 + 1)
            y_offset = np.random.randint(0, img_width - 224 + 1)
        crop = scaled_img[y_offset:y_offset + 224, x_offset:x_offset + 224, :].copy()
        # io.imsave('./experiments/train_classifier/' + str(iter) + '.jpg', crop)

        data_array = np.transpose(crop, (2, 0, 1))
        # result = np.zeros(len(self.categories))
        # result[class_idx] = 1
        result = class_idx

        iter += 1

        return torch.tensor(data_array, dtype=torch.float, device=self.device), \
               torch.tensor(result, dtype=torch.long, device=self.device)

    def __len__(self):
        return len(self.file_paths)


def getResiscData(train_proportion=0.8, device='cpu'):
    data_base = os.path.join(dataset_common_dir, 'NWPU-RESISC45')
    scenes = os.listdir(data_base)
    categories = {scenes[i]: i for i in range(len(scenes))}
    indices = np.arange(0, 700)
    np.random.shuffle(indices)
    train_files = []
    val_files = []
    for scene in scenes:
        img_files = os.listdir(os.path.join(data_base, scene))
        for i in range(int(indices.shape[0] * train_proportion)):
            img_file = img_files[indices[i]]
            train_files.append(os.path.join(scene, img_file))
        for i in range(int(indices.shape[0] * train_proportion), indices.shape[0]):
            img_file = img_files[indices[i]]
            val_files.append(os.path.join(scene, img_file))
    return ResiscDataset(train_files, device, categories), ResiscDataset(val_files, device, categories)


class RemoteDataReader:
    def __init__(self):
        self.__ids = []
        with open(os.path.join(data_rs_dir, 'ids'), 'r') as id_file:
            for line in id_file.readlines():
                self.__ids.append(line[:-1])
        self.__length = len(self.__ids)
        self.__next_id = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.__next_id == self.__length:
            raise StopIteration
        id = self.__ids[self.__next_id]
        map = io.imread(os.path.join(data_rs_dir, id + '.jpg'))
        query = io.imread(os.path.join(data_rs_dir, id + '_q.jpg'))
        self.__next_id += 1
        return id, map, query
