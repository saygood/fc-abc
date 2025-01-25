import torch
import torch.nn as nn
import torch.optim as optim
from model.googlenet import Inception
from math import log
import utils.common as utils
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import copy
import sys
import random
import numpy as np

import argparse
import ast
from data import cifar10, cifar100, imagenet
from importlib import import_module
from thop import profile

#Namespace(arch='vgg_cifar', bee_from_scratch=False, best_honey=None, best_honey_past=None, best_honey_s=None, calfitness_epoch=2, cfg='vgg16', data_path='/home/lmb/cvpr_vgg2/data', data_set='cifar10', eval_batch_size=256, food_dimension=13, food_limit=5, food_number=10, from_scratch=False, gpus=None, honey_model=None, honeychange_num=2, job_dir='experiments/', label_smooth=False, lr=0.1, lr_decay_step=30, max_cycle=10, max_preserve=9, momentum=0.9, num_epochs=150, preserve_type='layerwise', random_rule='default', refine=None, reset=False, resume=None, split_optimizer=False, test_only=False, train_batch_size=256, warm_up=False, weight_decay=0.0001)

parser = argparse.ArgumentParser(description='Prune via BeePruning')
parser.add_argument('--from_scratch', action='store_true', help='Train from scratch?')
parser.add_argument('--bee_from_scratch', action='store_true', help='Beepruning from scratch?')
parser.add_argument('--label_smooth', action='store_true', help='Use Lable smooth criterion?')
parser.add_argument('--split_optimizer', action='store_true', help='Split the weight parameter that need weight decay?')
parser.add_argument('--warm_up', action='store_true', help='Use warm up LR?')
parser.add_argument('--gpus', type=int, nargs='+', default=[1], help='Select gpu_id to use. default:[0]', )
parser.add_argument('--sr', type=float, default=0.0001, help='sparsity factor for sparsity training', )
parser.add_argument('--s', type=bool, default=False, help='the sparsity factor for sparsity training', )
parser.add_argument('--data_set', type=str, default='cifar10', help='Select dataset to train. default:cifar10', )
parser.add_argument('--data_path', type=str, default='./data',help='The dictionary where the input is stored. default:', )
parser.add_argument('--job_dir', type=str, default='./experiments/Training/resnet56_ding',help='The directory where the summaries will be stored. default:./experiments', )
parser.add_argument('--reset', action='store_true', help='reset the directory?')
parser.add_argument('--resume', type=str, default=None, help='Load the model from the specified checkpoint./home/chengyijie/pycharm/ABCPruner/experiments/Training/early_brid/checkpoint/model_best.pt')
parser.add_argument('--refine', type=str, default=None, help='Path to the model to be fine-tuned.')#

## Training
parser.add_argument('--arch', type=str, default='resnet_cifar', help='Architecture of model. default:vgg_cifar')
parser.add_argument('--cfg',type=str,default='resnet56',help='Detail architecuture of model. default:vgg16')
parser.add_argument('--num_epochs',type=int,default=150,help='The num of epochs to train. default:150')
parser.add_argument('--train_batch_size',type=int,default=64,help='Batch size for training. default:256')
parser.add_argument('--eval_batch_size',type=int,default=64,help='Batch size for validation. default:256')
parser.add_argument('--momentum',type=float,default=0.9,help='Momentum for MomentumOptimizer. default:0.9')
parser.add_argument('--lr',type=float,default=0.1, help='Learning rate for train. default:0.1')
parser.add_argument('--lr_decay_step',type=int,default=[50,100],help='the iterval of learn rate decay. default:30')
parser.add_argument('--weight_decay',type=float, default=5e-3, help='The weight decay of loss. default:1e-4')
parser.add_argument('--test_only',action='store_true',help='Test only?')
args = parser.parse_args()

checkpoint = utils.checkpoint(args)
print(args)
device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
loss_func = nn.CrossEntropyLoss()

# Data
print('==> Loading Data..')
if args.data_set == 'cifar10':
    loader = cifar10.Data(args)
elif args.data_set == 'cifar100':
    loader = cifar100.Data(args)
else:
    loader = imagenet.Data(args)

def updateBN(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.sr*torch.sign(m.weight.data))  # L1

# Training
def train(model, optimizer, trainLoader, args, epoch):

    model.train()
    losses = utils.AverageMeter()
    accurary = utils.AverageMeter()
    print_freq = len(trainLoader.dataset) // args.train_batch_size // 10
    start_time = time.time()
    for batch, (inputs, targets) in enumerate(trainLoader):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_func(output, targets)
        loss.backward()
        losses.update(loss.item(), inputs.size(0))
        if args.s:
            print("sparsity......")
            updateBN(model)
        optimizer.step()

        prec1 = utils.accuracy(output, targets)
        accurary.update(prec1[0], inputs.size(0))

        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            logger.info(
                'Epoch[{}] ({}/{}):\t'
                'Loss {:.4f}\t'
                'Accurary {:.2f}%\t\t'
                'Time {:.2f}s'.format(
                    epoch, batch * args.train_batch_size, len(trainLoader.dataset),
                    float(losses.avg), float(accurary.avg), cost_time
                )
            )
            start_time = current_time

#Testing
def test(model, testLoader):
    global best_acc
    model.eval()

    losses = utils.AverageMeter()
    accurary = utils.AverageMeter()
    start_time = time.time()
    #normal conv_bn

    #depwise conv_bn
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            predicted = utils.accuracy(outputs, targets)
            accurary.update(predicted[0], inputs.size(0))

        current_time = time.time()
        logger.info(
            'Test Loss {:.4f}\tAccurary {:.2f}%\t\tTime {:.2f}s\n'
            .format(float(losses.avg), float(accurary.avg), (current_time - start_time))
        )
    return accurary.avg


def e_dist(a, b, metric='e'):
    """Distance calculation for 1D, 2D and 3D points using einsum

    preprocessing :
        use `_view_`, `_new_view_` or `_reshape_` with structured/recarrays

    Parameters
    ----------
    a, b : array like
        Inputs, list, tuple, array in 1, 2 or 3D form
    metric : string
        euclidean ('e', 'eu'...), sqeuclidean ('s', 'sq'...),

    Notes
    -----
    mini e_dist for 2d points array and a single point

    See Also
    --------
    cartesian_dist : function
        Produces pairs of x,y coordinates and the distance, without duplicates.
    """
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

    # Model

    if args.arch == 'vgg_cifar':
        model = import_module(f'model.{args.arch}').VGG(vgg_name='vgg16').to(device)
        temp = [1, 4, 8, 11, 15, 18, 21, 25, 28, 31, 35, 38, 41]
    elif args.arch == 'resnet_cifar':
        model = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)
    elif args.arch == 'googlenet':
        model = import_module(f'model.{args.arch}').googlenet(honey=[]).to(device)
    elif args.arch == 'densenet':
        model = import_module(f'model.{args.arch}').densenet().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_step, gamma=0.1)

    if args.resume:
        resumeckpt = torch.load(args.resume)
        state_dict = resumeckpt['state_dict']
        if args.cfg == 'resnet110':
            pass
        else:
            optimizer.load_state_dict(resumeckpt['optimizer'])
            scheduler.load_state_dict(resumeckpt['scheduler'])
        model.load_state_dict(state_dict)
        test(model, loader.testLoader)

    if len(args.gpus) != 1:
        model = nn.DataParallel(model, device_ids=args.gpus)
    stru = []

    #vgg
    # for name, para in model.named_parameters():
    #     temp_name = name.split('.')
    #     if len(temp_name) == 3 and int(temp_name[1]) in temp:
    #         para.requires_grad = True
    #     else:
    #         para.requires_grad = False
    #resnet56
    # if args.cfg == 'resnet56':
    #     for name, para in model.named_parameters():
    #         temp_name = name.split('.')
    #         if len(temp_name) == 2:
    #             if 'bn' in temp_name[0]:
    #                 para.requires_grad = True
    #             else:
    #                 para.requires_grad = False
    #         elif len(temp_name) == 4:
    #             if 'bn' in temp_name[2]:
    #                 para.requires_grad = True
    #             else:
    #                 para.requires_grad = False

    #resnet110
    if args.cfg == 'resnet110':
        for name, para in model.named_parameters():
            temp_name = name.split('.')
            if len(temp_name) == 2:
                if 'bn' in temp_name[0]:
                    para.requires_grad = True
                else:
                    para.requires_grad = False
            elif len(temp_name) == 4:
                if 'bn' in temp_name[2]:
                    para.requires_grad = True
                else:
                    para.requires_grad = False

    for epoch in range(start_epoch, args.num_epochs):
        stru.append(impact_compute(model))
        train(model, optimizer, loader.trainLoader, args, epoch)
        scheduler.step()
        test_acc = test(model, loader.testLoader)

        is_best = best_acc < test_acc
        best_acc = max(best_acc, test_acc)

        model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()

        state = {
            'state_dict': model_state_dict,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1,
        }
        checkpoint.save_model(state, epoch + 1, is_best)
        # for name, param in model.named_parameters():
        #     print(name, param.requires_grad)
        logger.info('Best accurary: {:.3f}'.format(float(best_acc)))

    hitmap_m = e_dist(stru, stru)
    pmax = np.max(hitmap_m)
    pmin = np.min(hitmap_m)
    hitmap_m = (hitmap_m - pmin) / (pmax - pmin)
    # with open('./heatmap_normal.txt','a+',encoding='utf-8') as f:
    #     f.writelines(str(hitmap_m.tolist()))
    # plt.figure()
    sns.heatmap(hitmap_m, vmin=0, vmax=1, cmap='crest_r')

    # s1 = p1.get_figure()
    str_name = "heatmap_resnet110_cnn" + ".jpg"
    plt.savefig(str_name, dpi=400, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
#nohup /home/chengyijie/.conda/envs/chen_p38/bin/python3.7 -u /home/chengyijie/pycharm/ABCPruner/bee_cifar.py >/dev/null 2>&1 &