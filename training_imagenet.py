import torch
import torch.nn as nn
import torch.optim as optim
import utils.common as utils

import os
import copy
import time
import math
import sys
import numpy as np
import heapq
import random
from data import imagenet_dali
from importlib import import_module
import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import ast
import os


parser = argparse.ArgumentParser(description='Prune via BeePruning')
parser.add_argument('--from_scratch',action='store_true',help='Train from scratch?')
parser.add_argument('--bee_from_scratch',action='store_true',help='Beepruning from scratch?')
parser.add_argument('--label_smooth',action='store_true',help='Use Lable smooth criterion?')
parser.add_argument('--split_optimizer',action='store_true',help='Split the weight parameter that need weight decay?')
parser.add_argument('--warm_up',action='store_true',help='Use warm up LR?')
parser.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3],help='Select gpu_id to use. default:[0]',)
parser.add_argument('--sr',type=float,default=0.0001,help = 'sparsity factor for sparsity training',)
parser.add_argument('--s',type=bool,default=False,help = 'the sparsity factor for sparsity training',)
parser.add_argument('--data_set',type=str,default='imagenet',help='Select dataset to train. default:cifar10',)
parser.add_argument('--data_path',type=str,default='/home/chengyijie/imagenet/ILSVRC/Data/CLS-LOC',help='The dictionary where the input is stored. default:',)
parser.add_argument('--job_dir',type=str,default='./experiments/Training/resnet50_imagenet/finetune',help='The directory where the summaries will be stored. default:./experiments',)
parser.add_argument('--reset', action='store_true',help='reset the directory?')
parser.add_argument('--resume',type=str,default='/home/chengyijie/pycharm/ABCPruner/experiments/Training/resnet50_imagenet/checkpoint/model_90.pt',help='Load the model from the specified checkpoint.')#/home/chengyijie/pycharm/ABCPruner/data/resnet50.pth
parser.add_argument('--refine',type=str,default=None,help='Path to the model to be fine-tuned.')
## Training
parser.add_argument('--arch',type=str,default='resnet',help='Architecture of model. default:vgg_cifar')
parser.add_argument('--cfg',type=str,default='resnet50',help='Detail architecuture of model. default:vgg16')
parser.add_argument('--num_epochs',type=int,default=90,help='The num of epochs to train. default:150')
parser.add_argument('--train_batch_size',type=int,default=256,help='Batch size for training. default:256')
parser.add_argument('--eval_batch_size',type=int,default=256,help='Batch size for validation. default:256')
parser.add_argument('--momentum',type=float,default=0.9,help='Momentum for MomentumOptimizer. default:0.9')
parser.add_argument('--lr',type=float,default=0.1,help='Learning rate for train. default:0.1')
parser.add_argument('--lr_decay_step',type=int,default=[50, 100], help='the iterval of learn rate decay. default:30')
parser.add_argument('--weight_decay',type=float,default=0.0001, help='The weight decay of loss. default:1e-4')
parser.add_argument('--test_only',action='store_true',help='Test only?')
parser.add_argument('--preserve_type',type = str,default = 'layerwise',help = 'The preserve ratio of each layer or the preserve ratio of the entire network')
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
checkpoint = utils.checkpoint(args)
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
loss_func = nn.CrossEntropyLoss()

# Data
print('==> Preparing data..')
def get_data_set(type='train'):
    if type == 'train':
        return imagenet_dali.get_imagenet_iter_dali('train', args.data_path, args.train_batch_size,
                                                   num_threads=8, crop=224, device_id=args.gpus[0], num_gpus=1)
    elif type == 'val':
        return imagenet_dali.get_imagenet_iter_dali('val', args.data_path, args.eval_batch_size,
                                                   num_threads=8, crop=224, device_id=args.gpus[0], num_gpus=1)
    else:
        return imagenet_dali.get_imagenet_iter_dali('test', args.data_path, args.eval_batch_size,
                                                   num_threads=8, crop=224, device_id=args.gpus[0], num_gpus=1)
trainLoader = get_data_set('train')
testLoader = get_data_set('val')

def adjust_learning_rate(optimizer, epoch, step, len_epoch, args):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = args.lr * (0.1 ** factor)

    """Warmup"""
    if epoch < 5 and args.warm_up:
        lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)


   #print('epoch{}\tlr{}'.format(epoch,lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Training
def train(model, optimizer, trainLoader, args, epoch, topk=(1,)):

    model.train()
    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    top5_accuracy = utils.AverageMeter()
    print_freq = trainLoader._size // args.train_batch_size // 10
    start_time = time.time()
    #trainLoader = get_data_set('train')
    #i = 0
    for batch, batch_data in enumerate(trainLoader):
        #i+=1
        #if i>5:
            #break

        inputs = batch_data[0]['data'].to(device)

        targets = batch_data[0]['label'].squeeze().long().to(device)

        train_loader_len = int(math.ceil(trainLoader._size / args.train_batch_size))

        adjust_learning_rate(optimizer, epoch, batch, train_loader_len, args)


        output = model(inputs)
        loss = loss_func(output, targets)
        optimizer.zero_grad()
        loss.backward()
        losses.update(loss.item(), inputs.size(0))
        optimizer.step()

        prec1 = utils.accuracy(output, targets, topk=topk)
        accuracy.update(prec1[0], inputs.size(0))
        top5_accuracy.update(prec1[1], inputs.size(0))


        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            logger.info(
                'Epoch[{}] ({}/{}):\t'
                'Loss {:.4f}\t'
                'Top1 {:.2f}%\t'
                'Top5 {:.2f}%\t'
                'Time {:.2f}s'.format(
                    epoch, batch * args.train_batch_size, trainLoader._size,
                    float(losses.avg), float(accuracy.avg), float(top5_accuracy.avg), cost_time
                )
            )
            start_time = current_time
            

#Testing
def test(model, testLoader, topk=(1,)):
    model.eval()

    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    top5_accuracy = utils.AverageMeter()

    start_time = time.time()
    #testLoader = get_data_set('test')
    #i = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(testLoader):
            #i+=1
            #if i > 5:
                #break
            inputs = batch_data[0]['data'].to(device)
            targets = batch_data[0]['label'].squeeze().long().to(device)
            targets = targets.cuda(non_blocking=True)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            predicted = utils.accuracy(outputs, targets, topk=topk)
            accuracy.update(predicted[0], inputs.size(0))
            top5_accuracy.update(predicted[1], inputs.size(0))

        current_time = time.time()
        logger.info(
            'Test Loss {:.4f}\tTop1 {:.2f}%\tTop5 {:.2f}%\tTime {:.2f}s\n'
                .format(float(losses.avg), float(accuracy.avg), float(top5_accuracy.avg), (current_time - start_time))
        )

    return top5_accuracy.avg, accuracy.avg



def e_dist(a, b, metric='e'):

    a = np.array(a)
    b = np.atleast_2d(b)
    a_dim = a.ndim
    b_dim = b.ndim
    if a_dim == 1:
        a = a.reshape(1, 1, a.shape[0])
    if a_dim >= 2:
        a = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
    if b_dim > 2:
        b = b.reshape(np.prod(b.shape[:-1]), b.shape[-1])
    diff = a - b
    dist_arr = np.einsum('ijk,ijk->ij', diff, diff)
    if metric[:1] == 'e':
        dist_arr = np.sqrt(dist_arr)
    dist_arr = np.squeeze(dist_arr)
    return dist_arr


def impact_compute(model):
    model.eval()
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            rank_l = torch.zeros(size)
            temp = m.weight.data.abs().clone()
            st, in_t = torch.sort(temp)
            ti = 1
            for i in in_t:
                rank_l[i] = ti
                ti += 1
            bn[index:(index + size)] = rank_l
            index += size
    return bn.tolist()


def main():
    start_epoch = 0
    best_acc = 0.0
    best_acc_top1 = 0.0
    code = []
    #
    if args.arch == 'vgg':
        model = import_module(f'model.{args.arch}').VGG(num_classes=1000).to(device)
    elif args.arch == 'resnet':
        model = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)

    if len(args.gpus) != 1:
        model = nn.DataParallel(model, device_ids=args.gpus)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_step, gamma=0.1)
    print(device)
    stru = []
    if args.resume:
        print('=> Resuming from ckpt {}'.format(args.resume))
        ckpt = torch.load(args.resume)#, map_location=device
        new_ckpt = {}
        for key, value in ckpt['state_dict'].items():
            new_key = key
            if not key.startswith('module'):
                new_key = 'module.' + key
            new_ckpt[new_key] = value
        model.load_state_dict(new_ckpt)
        optimizer.load_state_dict(ckpt['optimizer'])
        # scheduler.load_state_dict(ckpt['scheduler'])
        print('=> Continue from epoch {}...'.format(start_epoch))
        test(model, testLoader, topk=(1, 5))
        testLoader.reset()


    if args.cfg == 'resnet50':
        for name, para in model.named_parameters():
            temp_name = name.split('.')
            if len(temp_name) == 3:
                if 'bn' in temp_name[1]:
                    para.requires_grad = True
                else:
                    para.requires_grad = False
            elif len(temp_name) == 5 or len(temp_name) == 6:
                if 'bn' in temp_name[3]:
                    para.requires_grad = True
                else:
                    para.requires_grad = False

    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    for epoch in range(start_epoch, args.num_epochs):
        stru.append(impact_compute(model))
        train(model, optimizer, trainLoader, args, epoch, topk=(1, 5))
        test_acc, test_acc_top1 = test(model, testLoader,topk=(1, 5))

        is_best = best_acc < test_acc
        best_acc_top1 = max(best_acc_top1, test_acc_top1)
        best_acc = max(best_acc, test_acc)

        model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()

        state = {
            'state_dict': model_state_dict,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            # 'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1,
            'honey_code': code
        }
        checkpoint.save_model(state, epoch + 1, is_best)
        trainLoader.reset()
        testLoader.reset()

        logger.info('Best accurary(top5): {:.3f} (top1): {:.3f}'.format(float(best_acc),float(best_acc_top1)))
    # hitmap_m = e_dist(stru, stru)
    # pmax = np.max(hitmap_m)
    # pmin = np.min(hitmap_m)
    # hitmap_m = (hitmap_m - pmin) / (pmax - pmin)
    # # with open('./heatmap_normal.txt','a+',encoding='utf-8') as f:
    # #     f.writelines(str(hitmap_m.tolist()))
    # # plt.figure()
    # sns.heatmap(hitmap_m, vmin=0, vmax=1, cmap='crest_r')
    #
    # # s1 = p1.get_figure()
    # str_name = "heatmap_resnet50_nocnn" + ".jpg"
    # plt.savefig(str_name, dpi=400, bbox_inches='tight')
    # plt.close()
if __name__ == '__main__':
    main()
