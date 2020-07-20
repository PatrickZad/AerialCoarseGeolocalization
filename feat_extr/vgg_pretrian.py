import os
import sys

proj_path = os.path.abspath('..')
sys.path.append(proj_path)

from common import *
from feat_extr import ExtractorFactory

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss

from .ignite_expand import ValPrecision, DataParaCheckModelCheckpoint

import logging

import torch
from torch.utils.data import DataLoader

from data.dataset import getResiscData

# config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class_num = 45
batch_size = 132
lr = 0.01
momentum = 0.9
l2_weight_decay = 5e-4
lr_factor = 0.1
max_epoch = 320
save_period = 5
log_period = 10
chang_lr_thred = 1e-4

global ITER
ITER = 0
global last_val_loss

data_para = torch.cuda.device_count() > 1


def retrain_classifier(extr_type, local_file=None):
    if 'bn' in extr_type:
        bn = True
    else:
        bn = False
    prefix = 'net' if bn else 'net_nobn'
    expr_out = os.path.join(proj_path, 'experiments', 'train_classifier')
    logging.basicConfig(filename=os.path.join(expr_out, 'train_log'),
                        level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('train_logger')

    train_dataset, val_dataset = getResiscData(device=device, train_proportion=0.8)
    feature_extractor = ExtractorFactory.create_feature_extractor(extr_type, device)

    net = feature_extractor.new_classifier(class_num)
    net.train()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    logger.info('Load trainset: ' + str(len(train_dataset)))
    logger.info('Load validset: ' + str(len(val_dataset)))
    logger.info('Start training.')

    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=l2_weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_factor, patience=5)

    if local_file is not None:
        dict2 = torch.load(local_file)
        optim_stat = dict2['optimizer']
        net_state = dict2['model']
        net.load_state_dict(net_state, False)
        net.to(device=device)
        optimizer.load_state_dict(optim_stat)

    loss = torch.nn.CrossEntropyLoss()

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    trainer = create_supervised_trainer(net, optimizer, loss, device=device)
    evaluator = create_supervised_evaluator(net, metrics={'loss': Loss(loss), 'precision': ValPrecision()},
                                            device=device)
    checkpointer = DataParaCheckModelCheckpoint(expr_out, prefix, n_saved=10, require_empty=False)

    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=save_period), checkpointer,
                              {'model': net, 'optimizer': optimizer})

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        global ITER
        ITER += 1
        if ITER % log_period == 0:
            logger.info("Epoch[{}] Batch[{}] Loss: {}".format(trainer.state.epoch, ITER, trainer.state.output))
            print("Epoch[{}] Batch[{}] Loss: {}".format(trainer.state.epoch, ITER, trainer.state.output))
        if len(train_loader) == ITER:
            ITER = 0

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):

        evaluator.run(val_loader)
        metrics = evaluator.state.metrics

        val_loss = metrics['loss']
        scheduler.step(val_loss)

        top1_count, top1_p = metrics['precision']

        logger.info("Validation Results - Epoch: {} Val_loss: {} Val_precision:count {} top1 {}"
                    .format(trainer.state.epoch, metrics['loss'], top1_count, top1_p))
        print("Validation Results - Epoch: {} Val_loss: {} Val_precision:count {} top1 {}"
              .format(trainer.state.epoch, metrics['loss'], top1_count, top1_p))

    trainer.run(train_loader, max_epochs=max_epoch)


if __name__ == '__main__':
    import feat_extr

    retrain_classifier(feat_extr.VGG, './model_zoo/checkpoints/net_nobn_checkpoint_3240.pth')
