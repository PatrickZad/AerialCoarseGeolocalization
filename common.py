import os
import numpy as  np

# os.environ['CUDA_VISIBLE_DEVICES']='2'
proj_path = os.getcwd()
print(proj_path)
base_dir = os.path.dirname(proj_path)
dataset_common_dir = os.path.join(base_dir, 'Datasets')
data_village_dir = os.path.join(dataset_common_dir, 'AerialGeolocalization', 'village','scaled')
data_gravel_dir = os.path.join(dataset_common_dir, 'AerialGeolocalization', 'gravel_pit','scaled')
expr_base = os.path.join(proj_path, 'experiments')


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
