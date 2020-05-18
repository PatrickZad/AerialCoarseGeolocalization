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
    def __init__(self, vgg16_obj=None, vgg16_file=None, device='cpu'):
        if vgg16_obj is not None:
            model = vgg16_obj
        elif vgg16_file is not None:
            model = vgg16_bn()
            with open(vgg16_file, 'rb') as file:
                model_state_dict = file.read()
            model.load_state_dict(model_state_dict, strict=False)
        else:
            model = vgg16_bn(pretrained=True)
        model.to(device=device)
        model.eval()
        self.__split_net(model)
        self.__max_poolinger = nn.MaxPool2d(kernel_size=2, stride=2)

    def __split_net(self, model):
        all_layers = list(model.children())
        self.__conv_part = all_layers[0]
        conv_layers = list(all_layers[0].children())
        self.__block1 = nn.Sequential(*conv_layers[:6])
        self.__block2 = nn.Sequential(*conv_layers[7:13])
        self.__block3 = nn.Sequential(*conv_layers[14:23])
        self.__block4 = nn.Sequential(*conv_layers[24:33])
        self.__block5 = nn.Sequential(*conv_layers[34:-1])

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
        return Classifier(self.conv_part, fc_part)

    def representations_of(self, input):

        r1 = self.__block1(input)
        m1 = self.__max_poolinger(r1)
        r2 = self.__block2(m1)
        m2 = self.__max_poolinger(r2)
        r3 = self.__block3(m2)
        m3 = self.__max_poolinger(r3)
        r4 = self.__block4(m3)
        m4 = self.__max_poolinger(r4)
        r5 = self.__block5(m4)
        m5 = self.__max_poolinger(r5)
        return (r1, r2, r3, r4, r5), (m1, m2, m3, m4, m5)
