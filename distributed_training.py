from common import *
from backbone.model import VGG16FeatureExtractor

import torch.multiprocessing as t_mp
import torch.nn as nn
import torch.distributed as dist

import logging

import torch
from torch.utils.data import DataLoader

from data.dataset import getResiscData

# config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class_num = 45
batch_size = 74
lr = 0.01
momentum = 0.9
l2_weight_decay = 5e-4
lr_factor = 0.1
max_epoch = 128
save_period = 10
log_period = 10
chang_lr_thred = 1e-5
global ITER
ITER = 0
global last_val_loss
last_val_loss = 0


def retrain_classifier(local_file=None, bn=True):
    expr_out = os.path.join(proj_path, 'experiments', 'train_classifier')
    logging.basicConfig(filename=os.path.join(expr_out, 'train_log'),
                        level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('train_logger')

    train_dataset, val_dataset = getResiscData(device=device)
    feature_extractor = VGG16FeatureExtractor(device=device, bn=bn)
    net = feature_extractor.new_classifier(class_num)
    net.train()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    logger.info('Load trainset: ' + str(len(train_dataset)))
    logger.info('Load validset: ' + str(len(val_dataset)))
    logger.info('Start training.')

    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=l2_weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_factor, patience=8)

    if local_file is not None:
        dict2 = torch.load(local_file)
        optim_stat = dict2['optimizer']  # ['state']
        net_state = dict2['model']
        net.load_state_dict(net_state, False)
        net.to(device=device)
        optimizer.load_state_dict(optim_stat)

    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(max_epoch):
        for i, (imgs, labels) in enumerate(train_loader):
            outputs = net(imgs)
            train_loss = loss_func(outputs, labels)

            optimizer.zero_grad()
            train_loss.backward()
            if (i + 1) % log_period == 0:
                logger.info("Epoch[{}] Batch[{}] Loss: {}".format(epoch + 1, i + 1, train_loss))
                print("Epoch[{}] Batch[{}] Loss: {}".format(epoch + 1, i + 1, train_loss))
        with net.eval():
            for imgs, labels in val_loader:
                outputs = net(imgs)
                loss = loss_func()
        scheduler.step()


if __name__ == '__main__':
    # retrain_classifier('model_zoo/checkpoints/net_bn_final.pth')
    retrain_classifier()
