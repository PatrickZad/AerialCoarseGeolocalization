from torchvision.models.vgg import vgg16_bn
import torch.nn as nn
import torch

torch.set_default_tensor_type(torch.DoubleTensor)


class Classifier(nn.Module):
    def __init__(self, conv_component, fc_component):
        super(Classifier, self).__init__()
        self.__conv = conv_component
        self.__fc = fc_component

    def forward(self, input):
        batch_size = input.shape[0]
        conv_feature = self.__conv(input)
        feature = torch.reshape(conv_feature, (batch_size, -1))
        out = self.__fc(feature)
        return out


class VGG16FeatureExtractor:
    def __init__(self, vgg16_obj=None, vgg16_file=None):
        if vgg16_obj is not None:
            self.__model = vgg16_obj
        elif vgg16_file is not None:
            self.__model = vgg16_bn()
            with open(vgg16_file, 'rb') as file:
                model_state_dict = file.read()
            self.__model.load_state_dict(model_state_dict)
        else:
            self.__model = vgg16_bn(pretrained=True)
        self.__split_blocks()

    def __split_blocks(self):
        all_layers = list(self.__model.children())
        conv_layers = list(all_layers[0].children())
        self.__block1 = nn.Sequential(*conv_layers[:7])
        self.__block2 = nn.Sequential(*conv_layers[7:14])
        self.__block3 = nn.Sequential(*conv_layers[14:24])
        self.__block4 = nn.Sequential(*conv_layers[24:34])
        self.__block5 = nn.Sequential(*conv_layers[34:])

    def new_classifier(self, class_num):
        fc_part = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, class_num),
            nn.Softmax(dim=1)
        )
        conv_part = nn.Sequential(
            self.__block1,
            self.__block2,
            self.__block3,
            self.__block4,
            self.__block5,
        )
        return Classifier(conv_part, fc_part)
