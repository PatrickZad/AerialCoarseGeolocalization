from ignite.metrics import Metric
from ignite.handlers import ModelCheckpoint
import torch
import torch.nn as nn


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
        mask = y == sorted[:, -1:]
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


class DataParaCheckModelCheckpoint(ModelCheckpoint):
    def __call__(self, engine, to_save):
        save_dict = {}
        for k, obj in self.to_save.items(nn.DataParallel, nn.parallel.DistributedDataParallel):
            if isinstance(obj, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                save_dict[k] = obj.module
            else:
                save_dict[k] = obj
        super(DataParaCheckModelCheckpoint, self).__call__(engine, save_dict)
