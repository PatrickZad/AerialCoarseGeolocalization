from torchvision.models.vgg import vgg16_bn, vgg16
import torch.nn as nn
import torch
import backbone
import abc


class VggExtractorCommon(backbone.FeatureExtractor):
    def new_classifier(self, class_num):
        fc_part = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, class_num),
            nn.Softmax(dim=1)
        )
        return backbone.PretrainClassifier(self._conv_part, fc_part)

    @abc.abstractmethod
    def _split_conv_blocks(self, conv_layers):
        pass

    def _split_net(self, model):
        all_layers = list(model.children())
        self._conv_part = all_layers[0]
        conv_layers = list(all_layers[0].children())
        self._locks = self._split_conv_blocks(conv_layers)

    def features_of(self, input_img):
        block_r = []
        max_r = []
        block_in = input
        for blk in self._blocks:
            r_blk = blk(block_in)
            r_arr = r_blk.detach().numpy()
            del r_blk
            r_t = torch.tensor(r_arr, dtype=torch.float, device=self._device)
            block_r.append(r_t)
            m_blk = nn.functional.max_pool2d(input=r_t, kernel_size=2, stride=2)
            max_r.append(m_blk)
            block_in = m_blk
        return block_r, max_r


class VggBasedFeatureExtractor(VggExtractorCommon):
    def _split_conv_blocks(self, conv_layers):
        return [nn.Sequential(*conv_layers[:4]), nn.Sequential(*conv_layers[5:9]),
                nn.Sequential(*conv_layers[10:16]), nn.Sequential(*conv_layers[17:23]),
                nn.Sequential(*conv_layers[24:-1])]


class VggBNBasedFeatureExtractor(VggExtractorCommon):
    def _split_conv_blocks(self, conv_layers):
        return [nn.Sequential(*conv_layers[:6]), nn.Sequential(*conv_layers[7:13]),
                nn.Sequential(*conv_layers[14:23]), nn.Sequential(*conv_layers[24:33]),
                nn.Sequential(*conv_layers[34:-1])]


class VggFeatureExtractor(VggBasedFeatureExtractor):
    def __init__(self, device):
        model = vgg16(pretrained=True)
        model.to(device=device)
        model.eval()
        self._device = device


class VggBNFeatureExtractor(VggBNBasedFeatureExtractor):
    def __init__(self, device):
        model = vgg16_bn(pretrained=True)
        model.to(device=device)
        model.eval()
        self._device = device


class VggTuneFeatureExtractor(VggBasedFeatureExtractor):
    def __init__(self, device):
        model = vgg16(pretrained=False)
        dict2 = torch.load(backbone.vgg_tune_file, map_location=device)
        net_state = dict2['model']
        model.load_state_dict(net_state, False)
        model.to(device=device)
        model.eval()
        self._device = device


class VggBNTuneFeatureExtractor(VggBNBasedFeatureExtractor):
    def __init__(self, device):
        model = vgg16_bn(pretrained=False)
        dict2 = torch.load(backbone.vgg_bn_tune_file, map_location=device)
        net_state = dict2['model']
        model.load_state_dict(net_state, False)
        model.to(device=device)
        model.eval()
        self._device = device


class Resnet50FeatureExtractor:
    pass


class Resnet50TuneFeatureExtractor:
    pass
