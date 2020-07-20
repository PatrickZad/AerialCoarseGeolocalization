import os
import numpy as  np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2 as cv

# os.environ['CUDA_VISIBLE_DEVICES']='2'
proj_path = os.getcwd()
print(proj_path)
base_dir = os.path.dirname(proj_path)
dataset_common_dir = os.path.join(base_dir, 'Datasets')
data_village_dir = os.path.join(dataset_common_dir, 'AerialGeolocalization', 'village', 'scaled')
data_gravel_dir = os.path.join(dataset_common_dir, 'AerialGeolocalization', 'gravel_pit', 'scaled')
expr_base = os.path.join(proj_path, 'experiments')
data_rs_dir = os.path.join(dataset_common_dir, 'AerialGeolocalization', 'remote')
model_dir = os.path.join(proj_path, 'model_zoo')


class ImgConnectedComponentUnionFind:
    def __init__(self, h, w):
        self.__array = np.zeros(shape=(h * w,)) - 1
        self.__w = w
        self.__h = h

    def find(self, x, y):
        assert y < self.__h
        assert x < self.__w
        idx = self.idx_of_coord(x, y)
        next = idx
        while self.__array[next] != -1:
            next = self.__array[next]
        return next

    def union(self, n1, n2):
        p_n1 = self.find(n1)
        p_n2 = self.find(n2)
        if p_n1 != p_n2:
            self.__array[n1] = n2

    def idx_of_coord(self, x, y):
        return (y - 1) * self.__w + x

    def coord_of_idx(self, idx):
        y = idx // self.__w + 1
        x = idx - (y - 1) * self.__w
        return x, y

    def prior_neighbors_of(self, x, y):
        pass


class LabelRecorder:
    def __init__(self):
        self.__save = {}
        self.__query = {}

    def add(self, label, coord):
        coord_key = str(coord[0]) + ',' + str(coord[1])
        label_key = str(label)
        if label_key not in self.__save.keys():
            self.__save[label_key] = [coord]
        self.__query[coord_key] = label

    def lable_of(self, coord):
        coord_key = str(coord[0]) + ',' + str(coord[1])
        return self.__query[coord_key]


def valid_prior_nerghbors(binary_img, coord):
    h, w = binary_img.shape
    assert 0 < coord[0] < w
    assert 0 < coord[1] < h
    result = []
    for i in range(max(0, coord[0] - 1), min(coord[0] + 2, w)):
        for j in range(max(0, coord[1] - 1), min(coord[1] + 2, h)):
            if binary_img[j][i] == 1:
                result.append((i, j))
    return result


def valid_neighbors(binary_img, center, searched_map):
    h, w = binary_img.shape
    assert 0 < center[0] < w
    assert 0 < center[1] < h
    result = []
    for i in range(max(0, center[0] - 1), min(center[0] + 2, w)):
        for j in range(max(0, center[1] - 1), min(center[1] + 2, h)):
            if binary_img[j][i] == 1 and searched_map[j][i] == 0:
                result.append((i, j))
    return result


'''def connected_components(binary_img):
    h, w = binary_img.shape
    label = 0
    # uf_set = ImgConnectedComponentUnionFind(h, w)
    recorder = LabelRecorder()
    for i in range(h):
        for j in range(w):
            valid_priors=valid_prior_nerghbors(binary_img,(j,i))
            if len(valid_priors)>0:
                pass
            else:
                recorder.add(label,(j,i))
                label+=1
'''


def connected_components(binary_img):
    h, w = binary_img.shape
    searched_map = np.zeros(shape=(h, w))
    components = []

    def search(binary_img, center, searched_map, mem_list):
        if searched_map[center[1]][center[0]] == 1:
            return
        neighbors = valid_neighbors(binary_img, center, searched_map)
        for pt in neighbors:
            mem_list.append(pt)
            searched_map[pt[1]][pt[0]] = 1
            search(binary_img, pt, searched_map, mem_list)

    for i in range(w):
        for j in range(h):
            if binary_img[j][i] == 1 and searched_map[j][i] == 0:
                comp_pts = [(i, j)]
                search(binary_img, (i, j), searched_map, comp_pts)
                components.append(comp_pts)
    return components


def save_as_heatmap(float_arr, filepath):
    float_01_img = float_arr / float_arr.max()
    int_img = np.uint8(float_01_img * 255)
    heatmap = cv.applyColorMap(int_img, cv.COLORMAP_JET)
    cv.imwrite(filepath, heatmap)


def default_corners(img):
    return np.array([[0, 0], [img.shape[1] - 1, 0], [img.shape[1] - 1, img.shape[0] - 1],
                     [0, img.shape[0] - 1]])


def adaptive_affine(img, affine_mat, content_corners=None):
    # 2 by 2 mat
    # auto translation
    if content_corners is None:
        content_corners = default_corners(img)
    affined_corners = np.int32(np.matmul(content_corners, affine_mat.T))
    x, y, w, h = cv.boundingRect(affined_corners)
    translation = np.array([-x, -y])
    for corner in affined_corners:
        corner += translation
    # return affined and translated corners,adaptive translation affine mat,bounding rectangular width and height
    affine_mat = np.concatenate([affine_mat, translation.reshape((2, 1))], axis=1)
    return affined_corners, affine_mat, (w, h)


def warp_pts(pts_array, homo_mat):
    # (x,y) coordinate
    homog_pts = np.concatenate([pts_array, np.ones((pts_array.shape[0], 1))], axis=-1)
    warp_homog_pts = np.matmul(homog_pts, homo_mat.T)
    warp_homog_pts /= warp_homog_pts[:, 2:]
    return warp_homog_pts[:, :-1]


def rotation_phi(img, phi, content_corners=None):
    if content_corners is None:
        content_corners = default_corners(img)
    if phi == 0:
        return img, np.concatenate([np.eye(2), np.zeros((2, 1))], axis=1), content_corners
    phi = np.deg2rad(phi)
    s, c = np.sin(phi), np.cos(phi)
    mat_rot = np.array([[c, -s], [s, c]])
    rot_corners, affine_mat, bounding = adaptive_affine(img, mat_rot, content_corners)
    affined = cv.warpAffine(img, affine_mat, bounding)
    return affined, affine_mat, rot_corners


def adaptive_scale(img, factor, content_corners=None):
    if content_corners is None:
        content_corners = default_corners(img)
    mat_scale = np.array([[factor, 0], [0, factor]])
    scale_corners, affine_mat, bounding = adaptive_affine(img, mat_scale, content_corners)
    scaled = cv.warpAffine(img, affine_mat, bounding)
    return scaled, affine_mat, scale_corners


def tilt_image(img, tilt, content_corners=None):
    if content_corners is None:
        content_corners = default_corners(img)
    if tilt == 1:
        return img, np.concatenate([np.eye(2), np.zeros((2, 1))], axis=1), content_corners
    gaussian_sigma = 0.8 * np.sqrt(tilt ** 2 - 1)
    unti_aliasing = cv.GaussianBlur(
        img, (3, 1), sigmaX=0, sigmaY=gaussian_sigma)
    mat_tilt = np.array([[1, 0], [0, 1 / tilt]])
    tilt_corners, affine_mat, bounding = adaptive_affine(img, mat_tilt, content_corners)
    affined = cv.warpAffine(unti_aliasing, affine_mat, bounding)
    return affined, affine_mat, tilt_corners
