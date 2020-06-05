from common import *
from backbone.model import VGG16FeatureExtractor

import torch.multiprocessing as t_mp
import torch.nn as nn
import torch.distributed as dist

import logging
import os

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
max_epoch = 320
save_period = 5
log_period = 10
chang_lr_thred = 1e-4
n_save = 10
# distributed training params
nodes = 1
gpus = 1
world_size = nodes * gpus
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '9601'
node_rank = 0


def retrain_classifier(gpu_id, local_file=None, bn=True, model_prefix='net'):
    rank = node_rank * gpus + gpu_id
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=world_size, rank=rank)

    expr_out = os.path.join(proj_path, 'experiments', 'train_classifier')
    logging.basicConfig(filename=os.path.join(expr_out, 'train_log'),
                        level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('train_logger_on_' + str(gpu_id))

    torch.cuda.set_device(gpu_id)

    train_dataset, val_dataset = getResiscData(device=device)
    feature_extractor = VGG16FeatureExtractor(device=device, bn=bn)
    net = feature_extractor.new_classifier(class_num)

    net.cuda(gpu_id)

    net.train()

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    logger.info('Load trainset: ' + str(len(train_dataset)))
    logger.info('Load validset: ' + str(len(val_dataset)))
    logger.info('Start training.')

    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=l2_weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_factor, patience=5)

    if local_file is not None:
        dict2 = torch.load(local_file)
        optim_stat = dict2['optimizer']  # ['state']
        net_state = dict2['model']
        net.load_state_dict(net_state, False)
        net.to(device=device)
        optimizer.load_state_dict(optim_stat)

    loss_func = torch.nn.CrossEntropyLoss().cuda(gpu_id)

    net = nn.parallel.DistributedDataParallel(net, device_ids=[gpu_id])

    val_loss_saved = np.array([], dtype=np.float)
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
            top1_correct = 0.0
            total_loss = 0
            for imgs, labels in val_loader:
                outputs = net(imgs)
                loss = loss_func()
                total_loss += loss * outputs.shape[0]
                # top1 count
                sorted = torch.argsort(outputs, dim=-1)
                top1_correct += (labels == sorted[:, 0]).sum()
            val_loss = total_loss / len(val_dataset)
            # TODO loss tensor to loss scalar
            logger.info("Validation Results - Epoch: {} Val_loss: {}\nVal_precision:count {}, top1 {}"
                        .format(epoch + 1, val_loss, top1_correct, top1_correct / len(val_dataset)))
            print("Validation Results - Epoch: {} Val_loss: {}\nVal_precision:count {}, top1 {}"
                  .format(epoch + 1, val_loss, top1_correct, top1_correct / len(val_dataset)))
        scheduler.step(val_loss)
        if (epoch + 1) % save_period == 0:
            # only in master proc
            if rank == 0:
                if val_loss_saved.shape[0] < n_save:
                    torch.save({'model': net.state_dict(), 'optimizaer': optimizer.state_dict()},
                               os.path.join(model_prefix + '_' + str(val_loss)[:8] + '.pth'))
                    val_loss_saved = np.concatenate([val_loss_saved, np.array([val_loss], dtype=np.float)], axis=0)
                else:
                    max_val = val_loss_saved.max()
                    max_idx = val_loss_saved.argmax()
                    if val_loss < max_val:
                        torch.save({'model': net.state_dict(), 'optimizaer': optimizer.state_dict()},
                                   os.path.join(model_prefix + '_' + str(val_loss)[:8] + '.pth'))
                        val_loss_saved[max_idx] = val_loss
                        os.remove(os.path.join(model_prefix + '_' + str(max_val)[:8] + '.pth'))


if __name__ == '__main__':
    # retrain_classifier('model_zoo/checkpoints/net_bn_final.pth')
    t_mp.spawn(retrain_classifier, nprocs=gpus, args=(None, False, 'net_nobn'))
    retrain_classifier()
