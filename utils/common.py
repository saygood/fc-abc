from __future__ import absolute_import
import datetime
import shutil
from pathlib import Path
import os 

import torch
import logging
import time
import torch.nn as nn
import torch.nn.functional as F
'''
#label smooth
class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss
'''
"""Computes and stores the average and current value"""

class CELossWithLabelSmoothing(nn.Module):
    def __init__(self, n_classes, smoothing=0):
        super().__init__()

        assert 0 <= smoothing <= 1, "The argument `smoothing` must be between 0 and 1!"

        self.n_classes = n_classes
        self.smoothing = smoothing
        
    def forward(self, pred, gt):
        if gt.ndim == 1:
            gt = torch.eye(self.n_classes, device=gt.device)[gt]
            return self(pred, gt)
        elif gt.ndim == 2:
            log_prob = F.log_softmax(pred, dim=1)
            ce_loss = -torch.sum(gt * log_prob, dim=1)
            loss = (1 - self.smoothing) * ce_loss
            loss += self.smoothing * -torch.sum(log_prob, dim=1)
            return torch.mean(loss)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TopKAccuracy(nn.Module):
    def __init__(self, k):
        super().__init__()

        self.k = k

    def forward(self, pred, gt):
        _, topk = torch.topk(pred, k=self.k, dim=1)
        corr = torch.eq(topk, gt.unsqueeze(1).repeat(1, self.k))
        acc = corr.sum(dim=1).float().mean().item()
        return acc

def ensure_path(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
 
from datetime import timedelta
def get_elapsed_time(start_time):
    return timedelta(seconds=round(time.time() - start_time))


def mkdir(path):
    if not os.path.isdir(path):
        mkdir(os.path.split(path)[0])  
    else:
        return
    os.mkdir(path)

'''Save model and record configurations'''
class checkpoint():
    def __init__(self, args):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        today = datetime.date.today()

        self.args = args
        self.job_dir = args.job_dir
        self.ckpt_dir = self.job_dir + '/checkpoint'
        self.run_dir = self.job_dir + '/run'

        if args.reset:
            os.system('rm -rf' + args.job_dir)

        def _make_dir(path):
            if not os.path.exists(path):
                #print("pathdonotexist")
                os.makedirs(path)

        _make_dir(self.job_dir)
        _make_dir(self.ckpt_dir)
        _make_dir(self.run_dir)

        config_dir = self.job_dir + '/config.txt'
        if not os.path.exists(config_dir):
            with open(config_dir, 'w') as f:
                f.write(now + '\n\n')
                for arg in vars(args):
                    f.write('{}: {}\n'.format(arg, getattr(args, arg)))
                f.write('\n')

    def save_model(self, state, epoch, is_best):
        save_path = f'{self.ckpt_dir}/model_{epoch}.pt'
        # print('=> Saving model to {}'.format(save_path))
        torch.save(state, save_path)
        if is_best:
            shutil.copyfile(save_path, f'{self.ckpt_dir}/model_best.pt')

    def save_model_j1(self, state, epoch, is_best, best_acc):
        save_path = f'{self.ckpt_dir}/model_best_%s.pt'%best_acc
        
        if is_best:
            torch.save(state, save_path)
            
    def save_sparsity_model(self, state, filename):
        save_path = f'{self.ckpt_dir}/model_{filename}.pt'
        # print('=> Saving model to {}'.format(save_path))
        torch.save(state, save_path)


    def save_honey_model(self, state):
        save_path = f'{self.ckpt_dir}/bestmodel_after_bee.pt'
        # print('=> Saving model to {}'.format(save_path))
        torch.save(state, save_path)


def get_logger(file_path):

    logger = logging.getLogger('gal')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger

"""Computes the precision@k for the specified values of k"""
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True) #返回output中的最大值及其索引
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0,keepdim=True)
            res.append(correct_k.mul_(100.0/batch_size))
        return res

