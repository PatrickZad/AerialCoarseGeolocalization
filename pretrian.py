from common import *
from backbone.model import VGG16FeatureExtractor

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Loss
import logging

import torch
from torch.utils.data import DataLoader

from data.dataset import getResiscData

# config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class_num = 45
batch_size = 56
lr = 0.01
momentum = 0.9
l2_weight_decay = 5e-4
lr_factor = 0.1
max_epoch = 128
save_period = 10
log_period = 10
chang_lr_thred = 1e-3
global ITER
ITER = 0
global last_val_loss
last_val_loss = 0


def retrain_classifier(local_file=None):
    expr_out = os.path.join(proj_path, 'experiments', 'train_classifier')
    logging.basicConfig(filename=os.path.join(expr_out, 'train_log'),
                        level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('train_logger')

    train_dataset, val_dataset = getResiscData(device=device)

    feature_extractor = VGG16FeatureExtractor(device=device)
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
        dict2=torch.load(local_file)
        optim_stat = dict2['optimizer']#['state']
        net_state = dict2['model']
        net.load_state_dict(net_state, False)
        net.to(device=device)
        optimizer.load_state_dict(optim_stat)

    loss = torch.nn.CrossEntropyLoss()

    trainer = create_supervised_trainer(net, optimizer, loss, device=device)
    evaluator = create_supervised_evaluator(net, metrics={'loss': Loss(loss)}, device=device)
    checkpointer = ModelCheckpoint(expr_out, 'net', n_saved=10, require_empty=False)

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
        global last_val_loss
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics

        val_loss = metrics['loss']
        scheduler.step(val_loss)

        logger.info("Validation Results - Epoch: {} Val_loss: {}"
                    .format(trainer.state.epoch, metrics['loss']))
        print("Validation Results - Epoch: {} Val_loss: {}"
              .format(trainer.state.epoch, metrics['loss']))

    trainer.run(train_loader, max_epochs=max_epoch)


if __name__ == '__main__':
    retrain_classifier('./experiments/train_classifier/net_checkpoint_47280.pth')
