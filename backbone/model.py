from torchvision.models.vgg import vgg16_bn
import torch.nn as nn
import torch

torch.set_default_tensor_type(torch.FloatTensor)


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
            model = vgg16_bn(pretrained=False)
            dict2 = torch.load(vgg16_file, map_location=device)
            net_state = dict2['model']
            model.load_state_dict(net_state, False)
        else:
            model = vgg16_bn(pretrained=True)
        model.to(device=device)
        model.eval()
        self.__split_net(model)
        self.__device = device
        # self.__max_poolinger = nn.MaxPool2d(kernel_size=2, stride=2)

    def __split_net(self, model):
        all_layers = list(model.children())
        self.__conv_part = all_layers[0]
        conv_layers = list(all_layers[0].children())
        self.__blocks = [nn.Sequential(*conv_layers[:6]), nn.Sequential(*conv_layers[7:13]),
                         nn.Sequential(*conv_layers[14:23]), nn.Sequential(*conv_layers[24:33]),
                         nn.Sequential(*conv_layers[34:-1])]

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
        return Classifier(self.__conv_part, fc_part)

    def representations_of(self, input,return_max=True):
        block_r=[]
        max_r=[]
        block_in=input
        for blk in self.__blocks:
            r_blk=blk(block_in)
            r_arr=r_blk.detach().numpy()
            del r_blk
            r_t=torch.tensor(r_arr, dtype=torch.float, device=self.__device)
            block_r.append(r_t)
            m_blk=nn.functional.max_pool2d(input=r_t, kernel_size=2, stride=2)
            ''' m_arr=m_blk.detach().numpy()
            del m_blk
            m_t=torch.tensor(m_arr, dtype=torch.float, device=self.__device)
            '''
            max_r.append(m_blk)
            block_in=m_blk
        ''' r1 = self.__block1(input)
        r1_arr = r1.detach().numpy()
        del r1
        r1_t = torch.tensor(r1_arr, dtype=torch.float, device=self.__device)
        m1 = nn.functional.max_pool2d(input=r1_t, kernel_size=2, stride=2)
        m1_arr = m1.detach().numpy()
        del m1
        m1_t = torch.tensor(m1_arr, dtype=torch.float, device=self.__device)
        r2 = self.__block2(m1_t)
        r2_arr = r2.detach().numpy()
        del r2
        r2_t = torch.tensor(r2_arr, dtype=torch.float, device=self.__device)
        # m2 = self.__max_poolinger(r2)
        m2 = nn.functional.max_pool2d(input=r2, kernel_size=2, stride=2)
        m2.requires_grad_(False)
        r3 = self.__block3(m2)
        r3.requires_grad_(False)
        # m3 = self.__max_poolinger(r3)
        m3 = nn.functional.max_pool2d(input=r3, kernel_size=2, stride=2)
        m3.requires_grad_(False)
        r4 = self.__block4(m3)
        r4.requires_grad_(False)
        # m4 = self.__max_poolinger(r4)
        m4 = nn.functional.max_pool2d(input=r4, kernel_size=2, stride=2)
        m4.requires_grad_(False)
        r5 = self.__block5(m4)
        r5.requires_grad_(False)
        # m5 = self.__max_poolinger(r5)
        m5 = nn.functional.max_pool2d(input=r5, kernel_size=2, stride=2)
        m5.requires_grad_(False)
        return (r1, r2, r3, r4, r5), (m1, m2, m3, m4, m5)'''
        if return_max:
            return block_r,max_r
        return block_r