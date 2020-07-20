import abc
from feat_extr.models import FeatureExtractor
import numpy as np
import torch


class FeatureBasedLocationDetector:
    def __init__(self, feature_extr, device):
        assert isinstance(feature_extr, FeatureExtractor)
        self._extr = feature_extr
        self._device = device

    @abc.abstractmethod
    def map_features(self, map_img):
        pass

    @abc.abstractmethod
    def loc_features(self, loc_img):
        pass

    @abc.abstractmethod
    def detect_location(self, map_img, loc_img, map_features=None, loc_features=None):
        pass

    def _img2tensorbatch(self, img):
        img_array = np.transpose(img, (2, 0, 1))
        img_tensor = torch.tensor(img_array / 255, dtype=torch.float, device=self._device)
        img_tensor = torch.unsqueeze(img_tensor, 0)
        return img_tensor
