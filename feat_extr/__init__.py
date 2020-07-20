import os
import common
import torch
import torch.nn as nn
import abc
from .models import VggFeatureExtractor, VggBNFeatureExtractor, VggTuneFeatureExtractor, \
    VggBNTuneFeatureExtractor, Resnet50FeatureExtractor, Resnet50TuneFeatureExtractor

torch.set_default_tensor_type(torch.FloatTensor)

os.environ['TORCH_HOME'] = common.model_dir
backbone_types = ['vgg', 'vgg_bn', 'vgg_tune', 'vgg_bn_tune', 'resnet50', 'resnet50_tune']
VGG = backbone_types[0]
VGG_BN = backbone_types[1]
VGG_TUNE = backbone_types[2]
VGG_BN_TUNE = backbone_types[3]
RESNET50=backbone_types[4]
type_dict = {backbone_types[0]: VggFeatureExtractor,
             backbone_types[1]: VggBNFeatureExtractor,
             backbone_types[2]: VggTuneFeatureExtractor,
             backbone_types[3]: VggBNTuneFeatureExtractor,
             backbone_types[4]: Resnet50FeatureExtractor,
             backbone_types[5]: Resnet50TuneFeatureExtractor}


class ExtractorFactory:
    @staticmethod
    def create_feature_extractor(extr_type, device='cpu' if not torch.cuda.is_available() else 'cuda'):
        if extr_type in type_dict.keys():
            return type_dict[extr_type](device)
        else:
            assert False, 'Wrong extractor type !'
