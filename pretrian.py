from common import *
from backbone.model import VGG16FeatureExtractor

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Loss, Metric

import logging

import torch
from torch.utils.data import DataLoader

from data.dataset import getResiscData

# config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class_num = 45
batch_size = 78
lr = 0.01
momentum = 0.9
l2_weight_decay = 5e-4
lr_factor = 0.1
max_epoch = 128
save_period = 5
log_period = 10
chang_lr_thred = 1e-5
global ITER
ITER = 0
global last_val_loss


class ValPrecision(Metric):
    def __init__(self):
        super(ValPrecision, self).__init__()
        self._count_all = 0.0
        self._count_top1 = 0
        self._count_top5 = 0
        self._count_top10 = 0

    def reset(self):
        self._count_all = 0
        self._count_top1 = 0
        self._count_top5 = 0
        self._count_top10 = 0

    def update(self, output):
        y_pred, y = output
        y = torch.unsqueeze(y, dim=1)
        batch_size = y_pred.shape[0]
        self._count_all += batch_size
        sorted = torch.argsort(y_pred, dim=-1)
        mask = y == sorted[:, :1]
        top1_correct = torch.unsqueeze(mask.sum(), dim=0).cpu().numpy()[0]
        self._count_top1 += top1_correct
        '''top5_correct = 0
        top10_crrect=0
        
        top5_correct=torch.logi
        self._count_top5 += top5_correct
        top10_correct = (y in sorted[:, :10]).sum()
        self._count_top10 += top10_correct'''

    def compute(self):
        '''return self._count_top1 / self._count_all, \
               self._count_top5 / self._count_all, \
               self._count_top10 / self._count_all'''

        return self._count_top1, self._count_top1 / self._count_all


def retrain_classifier(local_file=None, bn=True):
    expr_out = os.path.join(proj_path, 'experiments', 'train_classifier')
    logging.basicConfig(filename=os.path.join(expr_out, 'train_log'),
                        level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('train_logger')

    train_dataset, val_dataset = getResiscData(device=device, train_proportion=0.8)
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

    loss = torch.nn.CrossEntropyLoss()

    trainer = create_supervised_trainer(net, optimizer, loss, device=device)
    evaluator = create_supervised_evaluator(net, metrics={'loss': Loss(loss), 'precision': ValPrecision()},
                                            device=device)
    checkpointer = ModelCheckpoint(expr_out, 'net_nobn', n_saved=10, require_empty=False)

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
    # retrain_classifier('model_zoo/checkpoints/net_checkpoint_38880.pth',bn=False)
    retrain_classifier(bn=False)
