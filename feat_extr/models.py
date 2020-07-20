from torchvision.models.vgg import vgg16_bn, vgg16
from torchvision.models.resnet import resnet50
import torch.nn as nn
import torch
import abc
from common import model_dir
import os

vgg_tune_file = os.path.join(model_dir, 'checkpoints', 'vgg_tune.pth')
vgg_bn_tune_file = os.path.join(model_dir, 'checkpoints', 'vgg_bn_tune.pth')


class FeatureExtractor:
    @abc.abstractmethod
    def _split_net(self, model):
        pass

    @abc.abstractmethod
    def new_classifier(self, class_num):
        pass

    @abc.abstractmethod
    def features_of(self, input_img):
        pass


class PretrainClassifier(nn.Module):
    def __init__(self, conv_component, fc_component):
        super(PretrainClassifier, self).__init__()
        self._conv = conv_component
        self._fc = fc_component

    def forward(self, input_batch):
        batch_size = input_batch.shape[0]
        conv_feature = self._conv(input_batch)
        feature = torch.reshape(conv_feature, (batch_size, -1))
        out = self._fc(feature)
        return out


class ExtractorCommon(FeatureExtractor):
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
        return PretrainClassifier(self._conv_part, fc_part)

    @abc.abstractmethod
    def _split_conv_blocks(self, conv_layers):
        pass

    def _split_net(self, model):
        all_layers = list(model.children())
        self._conv_part = all_layers[0]
        conv_layers = list(all_layers[0].children())
        self._blocks = self._split_conv_blocks(conv_layers)

    def features_of(self, input_img):
        block_r = []
        max_r = []
        block_in = input_img
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


class BasedFeatureExtractor(ExtractorCommon):
    def _split_conv_blocks(self, conv_layers):
        return [nn.Sequential(*conv_layers[:4]), nn.Sequential(*conv_layers[5:9]),
                nn.Sequential(*conv_layers[10:16]), nn.Sequential(*conv_layers[17:23]),
                nn.Sequential(*conv_layers[24:-1])]


class BNBasedFeatureExtractor(ExtractorCommon):
    def _split_conv_blocks(self, conv_layers):
        return [nn.Sequential(*conv_layers[:6]), nn.Sequential(*conv_layers[7:13]),
                nn.Sequential(*conv_layers[14:23]), nn.Sequential(*conv_layers[24:33]),
                nn.Sequential(*conv_layers[34:-1])]


class VggFeatureExtractor(BasedFeatureExtractor):
    def __init__(self, device):
        model = vgg16(pretrained=True)
        model.to(device=device)
        model.eval()
        self._device = device
        self._downsamp = 32
        super(VggFeatureExtractor, self)._split_net(model)


class VggBNFeatureExtractor(BNBasedFeatureExtractor):
    def __init__(self, device):
        model = vgg16_bn(pretrained=True)
        model.to(device=device)
        model.eval()
        self._device = device
        self._downsamp = 32
        super(VggBNFeatureExtractor, self)._split_net(model)


class VggTuneFeatureExtractor(BasedFeatureExtractor):
    def __init__(self, device):
        model = vgg16(pretrained=False)
        dict2 = torch.load(vgg_tune_file, map_location=device)
        net_state = dict2['model']
        model.load_state_dict(net_state, False)
        model.to(device=device)
        model.eval()
        self._device = device
        self._downsamp = 32
        super(VggTuneFeatureExtractor, self)._split_net(model)


class VggBNTuneFeatureExtractor(BNBasedFeatureExtractor):
    def __init__(self, device):
        model = vgg16_bn(pretrained=False)
        dict2 = torch.load(vgg_bn_tune_file, map_location=device)
        net_state = dict2['model']
        model.load_state_dict(net_state, False)
        model.to(device=device)
        model.eval()
        self._device = device
        self._downsamp = 32
        super(VggBNTuneFeatureExtractor, self)._split_net(model)


class Resnet50FeatureExtractor(ExtractorCommon):
    def __init__(self, device):
        model = resnet50(pretrained=True)
        model.to(device=device)
        model.eval()
        self._device = device
        self._downsamp = 32
        self._split_net(model)

    def _split_net(self, model):
        conv1 = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        conv2 = model.layer1
        conv3 = model.layer2
        conv4 = model.layer3
        conv5 = model.layer4
        self._blocks = [conv1, conv2, conv3, conv4, conv5]
        self._conv_part = nn.Sequential(*(self._blocks + [model.avgpool]))

    def _split_conv_blocks(self, conv_layers):
        pass

    def new_classifier(self, class_num):
        fc = nn.Sequential(nn.Linear(512 * 4, class_num), nn.Softmax())
        return PretrainClassifier(self._conv_part, fc)

    def features_of(self, input_img):
        block_r = []
        max_r = []
        block_in = input_img
        for blk in self._blocks:
            r_blk = blk(block_in)
            r_arr = r_blk.detach().numpy()
            del r_blk
            r_t = torch.tensor(r_arr, dtype=torch.float, device=self._device)
            block_r.append(r_t)
            max_r.append(r_t)
            block_in = r_t
        return block_r, max_r


class Resnet50TuneFeatureExtractor:
    pass


if __name__ == '__main__':
    import feat_extr

    model = Resnet50FeatureExtractor(device='cpu')
