import torch
import torch.nn as nn
import torch.optim as optim
from model.googlenet import Inception
from math import log
import utils.common as utils
import argparse
import os
import time
import copy
import sys
import random
import numpy as np
import heapq
import json

from data import cifar10, cifar100, imagenet
from importlib import import_module
from thop import profile

# Namespace(arch='vgg_cifar', bee_from_scratch=False, best_honey=None, best_honey_past=None, best_honey_s=None, calfitness_epoch=2, cfg='vgg16', data_path='/home/lmb/cvpr_vgg2/data', data_set='cifar10', eval_batch_size=256, food_dimension=13, food_limit=5, food_number=10, from_scratch=False, gpus=None, honey_model=None, honeychange_num=2, job_dir='experiments/', label_smooth=False, lr=0.1, lr_decay_step=30, max_cycle=10, max_preserve=9, momentum=0.9, num_epochs=150, preserve_type='layerwise', random_rule='default', refine=None, reset=False, resume=None, split_optimizer=False, test_only=False, train_batch_size=256, warm_up=False, weight_decay=0.0001)

parser = argparse.ArgumentParser(description='Prune model on imagenet via BeePruning')
parser.add_argument('--from_scratch', action='store_true', help='Train from scratch?')
parser.add_argument('--bee_from_scratch', action='store_true', help='Beepruning from scratch?')
parser.add_argument('--label_smooth', action='store_true', help='Use Lable smooth criterion?')
parser.add_argument('--split_optimizer', action='store_true', help='Split the weight parameter that need weight decay?')
parser.add_argument('--warm_up', action='store_true', help='Use warm up LR?')
parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='Select gpu_id to use. default:[0]', )
parser.add_argument('--sr', type=float, default=0.0001, help='sparsity factor for sparsity training', )
parser.add_argument('--s', type=bool, default=False, help='the sparsity factor for sparsity training', )
parser.add_argument('--data_set', type=str, default='cifar10', help='Select dataset to train. default:cifar10', )
parser.add_argument('--data_path', type=str, default='./data',
                    help='The dictionary where the input is stored. default:', )
parser.add_argument('--job_dir', type=str, default='./experiments/cifar/resnet110/bee_search',
                    help='The directory where the summaries will be stored. default:./experiments', )
parser.add_argument('--reset', action='store_true', help='reset the directory?')
parser.add_argument('--resume', type=str, default=None, help='Load the model from the specified checkpoint.')
parser.add_argument('--refine', type=str, default=None, help='Path to the model to be fine-tuned.')

## Training
parser.add_argument('--arch', type=str, default='resnet_cifar', help='Architecture of model. default:vgg_cifar')
parser.add_argument('--cfg', type=str, default='resnet110', help='Detail architecuture of model. default:vgg16')
parser.add_argument('--num_epochs', type=int, default=520, help='The num of epochs to train. default:150')
parser.add_argument('--train_batch_size', type=int, default=64, help='Batch size for training. default:256')
parser.add_argument('--eval_batch_size', type=int, default=256, help='Batch size for validation. default:256')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for MomentumOptimizer. default:0.9')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for train. default:0.1')
parser.add_argument('--lr_decay_step', type=int, default=[50, 100], help='the iterval of learn rate decay. default:30')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='The weight decay of loss. default:1e-4')
parser.add_argument('--random_rule', type=str, default='random_pretrain', help='Weight initialization criterion after random clipping. default:default optional:default,random_pretrain,l1_pretrain')
parser.add_argument('--test_only', action='store_true', help='Test only?')

# Beepruning
parser.add_argument('--honey_model', type=str,
                    default='/home/chengyijie/pycharm/ABCPruner/experiments/Training/resnet110_cifar/cnn/checkpoint/model_best.pt',
                    help='Path to the model wait for Beepruning. default:None')
parser.add_argument('--calfitness_epoch', type=int, default=2,
                    help='Calculate fitness of honey source: training epochs. default:2')
parser.add_argument('--max_cycle', type=int, default=30, help='Search for best pruning plan times. default:10')
parser.add_argument('--max_preserve', type=int, default=5, help='Minimum percent of training per layer')
parser.add_argument('--pr', type=float, default=0.67, help='pruning ratio of model')
parser.add_argument('--preserve_type', type=str, default='layerwise',
                    help='The preserve ratio of each layer or the preserve ratio of the entire network')
parser.add_argument('--food_number', type=int, default=10, help='Food number')
parser.add_argument('--food_dimension', type=int, default=55,
                    help='Food dimension: num of conv layers. default: vgg16->13 conv layer to be pruned')
parser.add_argument('--food_scale', type=list, default=[], help='obtain the network structure. default:[]', )
parser.add_argument('--food_limit', type=int, default=5,
                    help='Beyond this limit, the bee has not been renewed to become a scout bee,default:5')
parser.add_argument('--honeychange_num', type=int, default=4,
                    help='Number of codes that the nectar source changes each time')
parser.add_argument('--best_honey', type=int, nargs='+', default=None,
                    help='If this hyper-parameter exists, skip bee-pruning and fine-tune from this training method')
parser.add_argument('--best_honey_s', type=str, default=None, help='Path to the best_honey')
parser.add_argument('--best_honey_past', type=int, nargs='+', default=None, )
args = parser.parse_args()

checkpoint = utils.checkpoint(args)
print(args)
device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
loss_func = nn.CrossEntropyLoss()

conv_num_cfg = {
    'vgg16': 13,
    'resnet56': 30,
    'resnet110': 55,
    'googlenet': 9,
    'densenet': 36,
}
food_scale_cfg ={
'resnet56': [16] + [16] * 9 + [32] * 9 + [64] * 9,
'resnet110': [16] + [16] * 18 + [32] * 18 + [64] * 18,
}

food_dimension = conv_num_cfg[args.cfg]
args.food_scale = food_scale_cfg[args.cfg]


# Data
print('==> Loading Data..')
if args.data_set == 'cifar10':
    loader = cifar10.Data(args)
elif args.data_set == 'cifar100':
    loader = cifar100.Data(args)
else:
    loader = imagenet.Data(args)

if args.data_set == 'cifar10':
    input_image_size = 32
elif args.data_set == 'imagenet':
    input_image_size = 224

# Model
print('==> Loading Model..')
if args.arch == 'vgg_cifar':
    origin_model = import_module(f'model.{args.arch}').VGG_ori(args.cfg).to(device)
elif args.arch == 'resnet_cifar':
    origin_model = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)
elif args.arch == 'googlenet':
    origin_model = import_module(f'model.{args.arch}').googlenet().to(device)
elif args.arch == 'densenet':
    origin_model = import_module(f'model.{args.arch}').densenet().to(device)

if args.honey_model is None or not os.path.exists(args.honey_model):
    raise ('Honey_model path should be exist!')

ckpt = torch.load(args.honey_model)
origin_model.load_state_dict(ckpt['state_dict'])
oristate_dict = origin_model.state_dict()

input = torch.randn(1, 3, input_image_size, input_image_size).to(device)
oriflops, oriparams = profile(origin_model, inputs=(input,))


# Define BeeGroup
class BeeGroup():
    """docstring for BeeGroup"""

    def __init__(self):
        super(BeeGroup, self).__init__()
        self.code = []  # size : num of conv layers value:{1,2,3,4,5,6,7,8,9}
        self.fitness = 0
        self.rfitness = 0
        self.trail = 0


# Initilize global element
best_honey = BeeGroup()
NectraSource = []
EmployedBee = []
OnLooker = []
best_honey_state = {}
vis_dict = []


# load pre-train params
def load_vgg_honey_model(model, random_rule):
    # print(ckpt['state_dict'])
    global oristate_dict
    state_dict = model.state_dict()
    last_select_index = None  # Conv index selected in the previous layer
    bn_scaling = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_scaling.append(list(m.weight.data.abs()))
    cb_index = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):

            oriweight = oristate_dict[name + '.weight']

            curweight = state_dict[name + '.weight']
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)

            if orifilter_num != currentfilter_num and (
                    random_rule == 'random_pretrain' or random_rule == 'l1_pretrain' or random_rule == 'l1_bn_pretrain'):

                select_num = currentfilter_num
                if random_rule == 'random_pretrain':
                    select_index = random.sample(range(0, orifilter_num - 1), select_num)
                    select_index.sort()
                elif random_rule == 'l1_pretrain':
                    l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                    select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                    select_index.sort()
                else:
                    l1_sum = bn_scaling[cb_index]
                    select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                    select_index.sort()
                    cb_index += 1

                if last_select_index is not None:
                    for index_i, i in enumerate(select_index):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name + '.weight'][index_i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                else:
                    for index_i, i in enumerate(select_index):
                        state_dict[name + '.weight'][index_i] = \
                            oristate_dict[name + '.weight'][i]

                last_select_index = select_index

            else:
                state_dict[name + '.weight'] = oriweight
                last_select_index = None
    model.load_state_dict(state_dict)


def load_resnet_honey_model(model, random_rule):
    cfg = {
        'resnet56': [9, 9, 9],
        'resnet110': [18, 18, 18],
    }

    global oristate_dict
    state_dict = model.state_dict()  #获取模型中的所有参数，包括可学习的参数和不可学习的参数

    current_cfg = cfg[args.cfg]
    last_select_index = None

    all_honey_conv_weight = []
    # for name, _ in model.named_modules():
    #     print(name)
    if args.cfg == 'resnet56' or args.cfg == 'resnet110':
        conv_name = 'conv1'
        conv_weight_name = conv_name + '.weight'
        all_honey_conv_weight.append(conv_weight_name)
        oriweight = oristate_dict[conv_weight_name]
        curweight = state_dict[conv_weight_name]
        orifilter_num = oriweight.size(0)
        currentfilter_num = curweight.size(0)
        if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

            select_num = currentfilter_num
            if random_rule == 'random_pretrain':
                select_index = random.sample(range(0, orifilter_num - 1), select_num)
                select_index.sort()
            else:
                l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                select_index.sort()
            last_select_index = select_index
            # logger.info('last_select_index{}'.format(last_select_index))

        else:
            state_dict[conv_weight_name] = oriweight
            last_select_index = None

    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            for l in range(2):
                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'
                all_honey_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)
                # logger.info('weight_num {}'.format(conv_weight_name))
                # logger.info('orifilter_num {}\tcurrentnum {}\n'.format(orifilter_num,currentfilter_num))
                # logger.info('orifilter  {}\tcurrent {}\n'.format(oristate_dict[conv_weight_name].size(),state_dict[conv_weight_name].size()))

                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    # print(select_num)
                    # print(orifilter_num)
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num - 1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()
                    if last_select_index is not None:
                        # logger.info('last_select_index'.format(last_select_index))
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index
                    # logger.info('last_select_index{}'.format(last_select_index))

                elif last_select_index != None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[conv_weight_name] = oriweight
                    last_select_index = None
    if args.cfg == 'resnet56' or args.cfg == 'resnet110':
        conv_name = 'fc'
        conv_weight_name = conv_name + '.weight'
        all_honey_conv_weight.append(conv_weight_name)
        oriweight = oristate_dict[conv_weight_name]
        orifilter_num = oriweight.size(0)
        if last_select_index != None:
            for index_i in range(orifilter_num):
                for index_j, j in enumerate(last_select_index):
                    state_dict[conv_weight_name][index_i][index_j] = \
                        oristate_dict[conv_weight_name][index_i][j]

        else:
            state_dict[conv_weight_name] = oriweight

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if conv_name not in all_honey_conv_weight:
                state_dict[conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            conv_name = name + '.weight'
            if conv_name not in all_honey_conv_weight:
                state_dict[name + '.weight'] = oristate_dict[name + '.weight']
                state_dict[name + '.bias'] = oristate_dict[name + '.bias']

    # for param_tensor in state_dict:
    # logger.info('param_tensor {}\tType {}\n'.format(param_tensor,state_dict[param_tensor].size()))
    # for param_tensor in model.state_dict():
    # logger.info('param_tensor {}\tType {}\n'.format(param_tensor,model.state_dict()[param_tensor].size()))
    # print(model)
    model.load_state_dict(state_dict)


# load pre-train params
def get_bn_score(honeysource):
    global origin_model
    global oristate_dict
    bn_score = 0.
    bn_scaling = []
    for m in origin_model.modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_scaling.append(m.weight.data.abs().cpu().numpy().tolist())

    for i in range(13):
        bn_score += sum(heapq.nlargest(
            int((honeysource[i] / 10 * 2 ** int(log(args.food_scale[i], 2) - log(args.food_scale[0], 2)))),
            bn_scaling[i]))
    return bn_score


def updateBN(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.sr * torch.sign(m.weight.data))  # L1


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
        if args.sr:
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


# Testing
def test(model, testLoader):
    global best_acc
    model.eval()

    losses = utils.AverageMeter()
    accurary = utils.AverageMeter()
    start_time = time.time()
    # depwise conv_bn
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


def impact_compute(model):
    '''
    ??????????????
    :return: ???????
    '''
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


def get_arch_flops(honey):
    if len(honey) > args.food_dimension:
        honey = honey[:args.food_dimension]
    if args.arch == 'vgg_cifar':
        model = import_module(f'model.{args.arch}').VGG_new(args.cfg, honeysource=honey, food_scale=args.food_scale).to(
            device)
    elif args.arch == 'resnet_cifar':
        model = import_module(f'model.{args.arch}').resnet(args.cfg, honey=honey).to(device)
    elif args.arch == 'googlenet':
        model = import_module(f'model.{args.arch}').googlenet(honey=honey).to(device)
    elif args.arch == 'densenet':
        model = import_module(f'model.{args.arch}').densenet(honey=honey).to(device)
    flops, params = profile(model, inputs=(input,))
    return flops


def calculationFitness(honey):
    global best_honey
    global vis_dict
    vis_dict.append(honey)
    fit_accurary = utils.AverageMeter()
    bn_score = get_bn_score(honey)
    fit_accurary.update(bn_score, 1)

    if fit_accurary.avg > best_honey.fitness:
        best_honey.code = copy.deepcopy(honey)
        best_honey.fitness = fit_accurary.avg

    return fit_accurary.avg


def calculationFitness(honey, train_loader, args):
    global best_honey
    global best_honey_state
    global vis_dict
    vis_dict.append(honey)
    # print(args.food_scale)
    if args.arch == 'resnet_cifar':
        model = import_module(f'model.{args.arch}').resnet_new(args.cfg, honey=honey).to(device)
        load_resnet_honey_model(model, args.random_rule)

    fit_accurary = utils.AverageMeter()
    train_accurary = utils.AverageMeter()

    # start_time = time.time()
    if len(args.gpus) != 1:
        model = nn.DataParallel(model, device_ids=args.gpus)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    model.train()
    for epoch in range(args.calfitness_epoch):
        for batch, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_func(output, targets)
            loss.backward()
            optimizer.step()
            prec1 = utils.accuracy(output, targets)
            train_accurary.update(prec1[0], inputs.size(0))

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader.testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predicted = utils.accuracy(outputs, targets)
            fit_accurary.update(predicted[0], inputs.size(0))

    '''
    logger.info(
            'Honey Source fintness {:.2f}%\t\tTime {:.2f}s\n'
            .format(float(accurary.avg), (current_time - start_time))
        )
    '''
    if fit_accurary.avg > best_honey.fitness:
        best_honey_state = copy.deepcopy(model.module.state_dict() if len(args.gpus) > 1 else model.state_dict())
        best_honey.code = copy.deepcopy(honey)
        best_honey.fitness = fit_accurary.avg

    return fit_accurary.avg


def change_depth(honey):
    param2change = np.random.randint(1, food_dimension - 1, args.honeychange_num)
    for j in range(args.honeychange_num):
        if honey[param2change[j]] - 1 > 0:
            honey[param2change[j]] -= 1
        else:
            honey[param2change[j]] = 1
    return honey


# Initilize Bee-Pruning
def initilize():
    print('==> Initilizing Honey_model..')
    global best_honey, NectraSource, EmployedBee, OnLooker

    honey_init = 0
    while honey_init < args.food_number:  # default:10
        NectraSource.append(copy.deepcopy(BeeGroup()))
        EmployedBee.append(copy.deepcopy(BeeGroup()))
        OnLooker.append(copy.deepcopy(BeeGroup()))
        honey = [random.randint(1, args.max_preserve * 2 ** (log(args.food_scale[j], 2) - log(args.food_scale[0], 2))) for j in range(food_dimension)]
        honey[0] = random.randint(1, 8)
        while True:
            if not is_legal(honey):
                change_depth(honey)
                continue
            else:
                NectraSource[honey_init].code = copy.deepcopy(honey)
                break
        print(NectraSource[honey_init].code)
        # initilize honey souce
        NectraSource[honey_init].fitness = calculationFitness(NectraSource[honey_init].code, loader.trainLoader, args)
        NectraSource[honey_init].rfitness = 0
        NectraSource[honey_init].trail = 0

        # initilize employed bee
        EmployedBee[honey_init].code = copy.deepcopy(NectraSource[honey_init].code)
        EmployedBee[honey_init].fitness = NectraSource[honey_init].fitness
        EmployedBee[honey_init].rfitness = NectraSource[honey_init].rfitness
        EmployedBee[honey_init].trail = NectraSource[honey_init].trail

        # initilize onlooker
        OnLooker[honey_init].code = copy.deepcopy(NectraSource[honey_init].code)
        OnLooker[honey_init].fitness = NectraSource[honey_init].fitness
        OnLooker[honey_init].rfitness = NectraSource[honey_init].rfitness
        OnLooker[honey_init].trail = NectraSource[honey_init].trail
        honey_init += 1
    # initilize best honey
    best_honey.code = copy.deepcopy(NectraSource[0].code)
    best_honey.fitness = NectraSource[0].fitness
    best_honey.rfitness = NectraSource[0].rfitness
    best_honey.trail = NectraSource[0].trail


# Send employed bees to find better honey source
def sendEmployedBees():
    global NectraSource, EmployedBee
    for i in range(args.food_number):

        while 1:
            k = random.randint(0, args.food_number - 1)
            if k != i:
                break

        EmployedBee[i].code = copy.deepcopy(NectraSource[i].code)
        Flag = True
        honeychange_num = args.honeychange_num
        while Flag:
            param2change = np.random.randint(0, food_dimension - 1, honeychange_num)
            if is_legal(EmployedBee[i].code):
                R = np.random.uniform(-0.5, 0.5, honeychange_num)
            else:
                R = np.random.uniform(-1, 0, honeychange_num)
            for j in range(honeychange_num):
                EmployedBee[i].code[param2change[j]] = int(NectraSource[i].code[param2change[j]] + R[j] * (NectraSource[i].code[param2change[j]] - NectraSource[k].code[param2change[j]]))
                if EmployedBee[i].code[param2change[j]] < 1:
                    EmployedBee[i].code[param2change[j]] = 1
                if OnLooker[i].code[0] > 8:
                    OnLooker[i].code[0] = 8
                elif param2change[j] != 0 and OnLooker[i].code[param2change[j]] > args.max_preserve * 2 ** (log(args.food_scale[param2change[j]], 2) - log(args.food_scale[0], 2)):
                    OnLooker[i].code[param2change[j]] = int(args.max_preserve * 2 ** (log(args.food_scale[param2change[j]], 2) - log(args.food_scale[0], 2)))
            if is_legal(EmployedBee[i].code):
                break
            else:
                honeychange_num -= 1
                if honeychange_num <= 0:
                    honeychange_num = 1
        EmployedBee[i].fitness = calculationFitness(EmployedBee[i].code, loader.trainLoader, args)

        if EmployedBee[i].fitness > NectraSource[i].fitness:
            NectraSource[i].code = copy.deepcopy(EmployedBee[i].code)
            NectraSource[i].trail = 0
            NectraSource[i].fitness = EmployedBee[i].fitness

        else:
            NectraSource[i].trail = NectraSource[i].trail + 1


# data normalization
def normalized_d(list):
    y = []
    maxl = max(list)
    minl = min(list)
    for i in list:
        y.append((i - minl) / (maxl - minl))
    return y


# Calculate whether a Onlooker to update a honey source
def calculateProbabilities():
    global NectraSource

    maxfit = NectraSource[0].fitness

    for i in range(1, args.food_number):
        if NectraSource[i].fitness > maxfit:
            maxfit = NectraSource[i].fitness
    for i in range(args.food_number):
        NectraSource[i].rfitness = (0.9 * (NectraSource[i].fitness / maxfit)) + 0.1


# Send Onlooker bees to find better honey source
def sendOnlookerBees():
    global NectraSource, EmployedBee, OnLooker
    i = 0
    t = 0
    while t < args.food_number:
        R_choosed = random.uniform(0, 1)
        if (R_choosed < NectraSource[i].rfitness):
            t += 1
            while 1:
                k = random.randint(0, args.food_number - 1)
                if k != i:
                    break
            OnLooker[i].code = copy.deepcopy(NectraSource[i].code)
            honeychange_num = args.honeychange_num
            while True:
                param2change = np.random.randint(0, food_dimension - 1, honeychange_num)
                if is_legal(OnLooker[i].code):
                    R = np.random.uniform(-0.5, 0.5, honeychange_num)
                else:
                    R = np.random.uniform(-1, 0, honeychange_num)

                for j in range(honeychange_num):
                    OnLooker[i].code[param2change[j]] = int(NectraSource[i].code[param2change[j]] + R[j] * (
                                NectraSource[i].code[param2change[j]] - NectraSource[k].code[param2change[j]]))
                    if OnLooker[i].code[param2change[j]] < 1:
                        OnLooker[i].code[param2change[j]] = 1
                    if OnLooker[i].code[0] > 8:
                        OnLooker[i].code[0] = 8
                    elif param2change[j] != 0 and OnLooker[i].code[param2change[j]] > args.max_preserve * 2 ** (log(args.food_scale[param2change[j]], 2) - log(args.food_scale[0], 2)):
                        OnLooker[i].code[param2change[j]] = int(args.max_preserve * 2 ** (log(args.food_scale[param2change[j]], 2) - log(args.food_scale[0], 2)))
                if is_legal(OnLooker[i].code):
                    break
                else:
                    honeychange_num -= 1
                    if honeychange_num <= 0:
                        honeychange_num = 1

            OnLooker[i].fitness = calculationFitness(OnLooker[i].code, loader.trainLoader, args)

            if OnLooker[i].fitness > NectraSource[i].fitness:
                NectraSource[i].code = copy.deepcopy(OnLooker[i].code)
                NectraSource[i].trail = 0
                NectraSource[i].fitness = OnLooker[i].fitness
            else:
                NectraSource[i].trail = NectraSource[i].trail + 1
        i += 1
        if i == args.food_number:
            i = 0


# If a honey source has not been update for args.food_limiet times, send a scout bee to regenerate it
def sendScoutBees():
    global NectraSource, EmployedBee, OnLooker
    maxtrailindex = 0
    for i in range(args.food_number):
        if NectraSource[i].trail > NectraSource[maxtrailindex].trail:
            maxtrailindex = i
    if NectraSource[maxtrailindex].trail >= args.food_limit:
        honey = [random.randint(1, args.max_preserve * 2 ** (log(args.food_scale[j], 2) - log(args.food_scale[0], 2)))
                 for j in range(food_dimension)]
        honey[0] = random.randint(1, 8)
        while True:
            if not is_legal(honey):
                change_depth(honey)
                continue
            else:
                NectraSource[maxtrailindex].code = copy.deepcopy(honey)
                break
        NectraSource[maxtrailindex].trail = 0
        NectraSource[maxtrailindex].fitness = calculationFitness(NectraSource[maxtrailindex].code, loader.trainLoader, args)


# Memorize best honey source
def memorizeBestSource():
    global best_honey, NectraSource
    for i in range(args.food_number):
        if NectraSource[i].fitness > best_honey.fitness:
            # print(NectraSource[i].fitness, NectraSource[i].code)
            # print(best_honey.fitness, best_honey.code)
            best_honey.code = copy.deepcopy(NectraSource[i].code)
            best_honey.fitness = NectraSource[i].fitness


def get_vgg_honey_state(model, random_rule):
    global oristate_dict
    state_dict = model.state_dict()
    last_select_index = None  # Conv index selected in the previous layer
    bn_scaling = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_scaling.append(list(m.weight.data.abs()))
    cb_index = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            oriweight = oristate_dict[name.replace('features','feature') + '.weight']
            curweight = state_dict[name + '.weight']
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)

            if orifilter_num != currentfilter_num and (
                    random_rule == 'random_pretrain' or random_rule == 'l1_pretrain' or random_rule == 'l1_bn_pretrain'):

                select_num = currentfilter_num
                if random_rule == 'random_pretrain':
                    select_index = random.sample(range(0, orifilter_num - 1), select_num)
                    select_index.sort()
                elif random_rule == 'l1_pretrain':
                    l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                    select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                    select_index.sort()
                else:
                    l1_sum = bn_scaling[cb_index]
                    select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                    select_index.sort()
                    cb_index += 1

                if last_select_index is not None:
                    for index_i, i in enumerate(select_index):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name + '.weight'][index_i][index_j] = \
                                oristate_dict[name.replace('features','feature') + '.weight'][i][j]
                else:
                    for index_i, i in enumerate(select_index):
                        state_dict[name + '.weight'][index_i] = \
                            oristate_dict[name.replace('features','feature') + '.weight'][i]

                last_select_index = select_index

            else:
                state_dict[name + '.weight'] = oriweight
                last_select_index = None
    model.load_state_dict(state_dict)


def get_dense_honey_model(model, random_rule):
    global oristate_dict

    state_dict = model.state_dict()

    conv_weight = []
    conv_trans_weight = []
    bn_weight = []
    bn_bias = []

    for i in range(3):
        for j in range(12):
            conv1_weight_name = 'dense%d.%d.conv1.weight' % (i + 1, j)
            conv_weight.append(conv1_weight_name)

            bn1_weight_name = 'dense%d.%d.bn1.weight' % (i + 1, j)
            bn_weight.append(bn1_weight_name)

            bn1_bias_name = 'dense%d.%d.bn1.bias' % (i + 1, j)
            bn_bias.append(bn1_bias_name)

    for i in range(2):
        conv1_weight_name = 'trans%d.conv1.weight' % (i + 1)
        conv_weight.append(conv1_weight_name)
        conv_trans_weight.append(conv1_weight_name)

        bn_weight_name = 'trans%d.bn1.weight' % (i + 1)
        bn_weight.append(bn_weight_name)

        bn_bias_name = 'trans%d.bn1.bias' % (i + 1)
        bn_bias.append(bn_bias_name)

    bn_weight.append('bn.weight')
    bn_bias.append('bn.bias')

    for k in range(len(conv_weight)):
        conv_weight_name = conv_weight[k]
        oriweight = oristate_dict[conv_weight_name]
        curweight = state_dict[conv_weight_name]
        orifilter_num = oriweight.size(1)
        currentfilter_num = curweight.size(1)
        select_num = currentfilter_num
        # print(orifilter_num)
        # print(currentfilter_num)

        if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):
            if random_rule == 'random_pretrain':
                select_index = random.sample(range(0, orifilter_num - 1), select_num)
                select_index.sort()
            else:
                l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                select_index.sort()

            for i in range(curweight.size(0)):
                for index_j, j in enumerate(select_index):
                    state_dict[conv_weight_name][i][index_j] = \
                        oristate_dict[conv_weight_name][i][j]

    for k in range(len(bn_weight)):

        bn_weight_name = bn_weight[k]
        bn_bias_name = bn_bias[k]
        bn_bias.append(bn_bias_name)
        bn_weight.append(bn_weight_name)
        oriweight = oristate_dict[bn_weight_name]
        curweight = state_dict[bn_weight_name]

        orifilter_num = oriweight.size(0)
        currentfilter_num = curweight.size(0)
        select_num = currentfilter_num

        if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):
            if random_rule == 'random_pretrain':
                select_index = random.sample(range(0, orifilter_num - 1), select_num)
                select_index.sort()
            else:
                l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                select_index.sort()

            for index_j, j in enumerate(select_index):
                state_dict[bn_weight_name][index_j] = \
                    oristate_dict[bn_weight_name][j]
                state_dict[bn_bias_name][index_j] = \
                    oristate_dict[bn_bias_name][j]

    oriweight = oristate_dict['fc.weight']
    curweight = state_dict['fc.weight']
    orifilter_num = oriweight.size(1)
    currentfilter_num = curweight.size(1)
    select_num = currentfilter_num

    if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):
        if random_rule == 'random_pretrain':
            select_index = random.sample(range(0, orifilter_num - 1), select_num)
            select_index.sort()
        else:
            l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
            select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
            select_index.sort()

        for i in range(curweight.size(0)):
            for index_j, j in enumerate(select_index):
                state_dict['fc.weight'][i][index_j] = \
                    oristate_dict['fc.weight'][i][j]

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if conv_name not in conv_weight:
                state_dict[conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.BatchNorm2d):
            bn_weight_name = name + '.weight'
            bn_bias_name = name + '.bias'
            if bn_weight_name not in bn_weight and bn_bias_name not in bn_bias:
                state_dict[bn_weight_name] = oristate_dict[bn_weight_name]
                state_dict[bn_bias_name] = oristate_dict[bn_bias_name]

    model.load_state_dict(state_dict)


def get_google_honey_model(model, random_rule):
    global oristate_dict
    state_dict = model.state_dict()

    last_select_index = None
    all_honey_conv_name = []
    all_honey_bn_name = []

    for name, module in model.named_modules():

        if isinstance(module, Inception):

            honey_filter_channel_index = ['.branch5x5.3']  # the index of sketch filter and channel weight
            honey_channel_index = ['.branch3x3.3', '.branch5x5.6']  # the index of sketch channel weight
            honey_filter_index = ['.branch3x3.0', '.branch5x5.0']  # the index of sketch filter weight
            honey_bn_index = ['.branch3x3.1', '.branch5x5.1', '.branch5x5.4']  # the index of sketch bn weight

            for bn_index in honey_bn_index:
                all_honey_bn_name.append(name + bn_index)

            for weight_index in honey_filter_channel_index:
                last_select_index = None
                conv_name = name + weight_index + '.weight'
                all_honey_conv_name.append(name + weight_index)

                oriweight = oristate_dict[conv_name]
                curweight = state_dict[conv_name]

                orifilter_num = oriweight.size(1)
                currentfilter_num = curweight.size(1)

                if orifilter_num != currentfilter_num and (
                        random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num - 1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()

                    # print(state_dict[conv_name].size())
                    # print(oristate_dict[conv_name].size())

                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                select_index_1 = copy.deepcopy(select_index)

                if orifilter_num != currentfilter_num and (
                        random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num - 1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()

                for index_i, i in enumerate(select_index):
                    for index_j, j in enumerate(select_index_1):
                        state_dict[conv_name][index_i][index_j] = \
                            oristate_dict[conv_name][i][j]

            for weight_index in honey_channel_index:

                conv_name = name + weight_index + '.weight'
                all_honey_conv_name.append(name + weight_index)

                oriweight = oristate_dict[conv_name]
                curweight = state_dict[conv_name]
                orifilter_num = oriweight.size(1)
                currentfilter_num = curweight.size(1)

                # print(state_dict[conv_name].size())
                # print(oristate_dict[conv_name].size())

                if orifilter_num != currentfilter_num and (
                        random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num - 1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()

                    for i in range(state_dict[conv_name].size(0)):
                        for index_j, j in enumerate(select_index):
                            state_dict[conv_name][i][index_j] = \
                                oristate_dict[conv_name][i][j]

            for weight_index in honey_filter_index:

                conv_name = name + weight_index + '.weight'
                all_honey_conv_name.append(name + weight_index)
                oriweight = oristate_dict[conv_name]
                curweight = state_dict[conv_name]

                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num and (
                        random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num - 1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()

                    for index_i, i in enumerate(select_index):
                        state_dict[conv_name][index_i] = \
                            oristate_dict[conv_name][i]

    for name, module in model.named_modules():  # Reassign non sketch weights to the new network

        if isinstance(module, nn.Conv2d):

            if name not in all_honey_conv_name:
                state_dict[name + '.weight'] = oristate_dict[name + '.weight']
                state_dict[name + '.bias'] = oristate_dict[name + '.bias']

        elif isinstance(module, nn.BatchNorm2d):

            if name not in all_honey_bn_name:
                state_dict[name + '.weight'] = oristate_dict[name + '.weight']
                state_dict[name + '.bias'] = oristate_dict[name + '.bias']
                state_dict[name + '.running_mean'] = oristate_dict[name + '.running_mean']
                state_dict[name + '.running_var'] = oristate_dict[name + '.running_var']

        elif isinstance(module, nn.Linear):
            state_dict[name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)


def get_resnet_honey_model(model, random_rule):
    cfg = {
        'resnet56': [9, 9, 9],
        'resnet110': [18, 18, 18],
    }

    global oristate_dict
    state_dict = model.state_dict()  # ???????????????????????????

    current_cfg = cfg[args.cfg]
    last_select_index = None

    all_honey_conv_weight = []

    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            for l in range(2):
                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'
                all_honey_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)
                # logger.info('weight_num {}'.format(conv_weight_name))
                # logger.info('orifilter_num {}\tcurrentnum {}\n'.format(orifilter_num,currentfilter_num))
                # logger.info('orifilter  {}\tcurrent {}\n'.format(oristate_dict[conv_weight_name].size(),state_dict[conv_weight_name].size()))

                if orifilter_num != currentfilter_num and (
                        random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num - 1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()
                    if last_select_index is not None:
                        logger.info('last_select_index'.format(last_select_index))
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index
                    # logger.info('last_select_index{}'.format(last_select_index))

                elif last_select_index != None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if conv_name not in all_honey_conv_weight:
                state_dict[conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name + '.bias'] = oristate_dict[name + '.bias']

    # for param_tensor in state_dict:
    # logger.info('param_tensor {}\tType {}\n'.format(param_tensor,state_dict[param_tensor].size()))
    # for param_tensor in model.state_dict():
    # logger.info('param_tensor {}\tType {}\n'.format(param_tensor,model.state_dict()[param_tensor].size()))

    model.load_state_dict(state_dict)


def is_legal(honey):
    global vis_dict
    if honey in vis_dict:
        return False

    flops = get_arch_flops(honey)
    if flops / oriflops > (1 - args.pr):
        # print('flops limit exceed')
        return False
    return True


def main():
    start_epoch = 0
    best_acc = 0.0
    code = []

    if args.resume == None:

        test(origin_model, loader.testLoader)

        if args.best_honey == None:  # best_honey

            start_time = time.time()

            bee_start_time = time.time()

            print('==> Start BeePruning..')

            initilize()

            # memorizeBestSource()

            for cycle in range(args.max_cycle):  # max_cycle:50

                current_time = time.time()
                logger.info(
                    'Search Cycle [{}]\t Best Honey Source {}\tBest Honey Source fitness {:.2f}%\tTime {:.2f}s\n'
                        .format(cycle, best_honey.code, float(best_honey.fitness), (current_time - start_time))
                )
                start_time = time.time()

                sendEmployedBees()

                calculateProbabilities()

                sendOnlookerBees()

                # memorizeBestSource()

                sendScoutBees()

                # memorizeBestSource()

            print('==> BeePruning Complete!')
            bee_end_time = time.time()
            logger.info(
                'Best Honey Source {}\tBest Honey Source fitness {:.2f}%\tTime Used{:.2f}s\n'
                    .format(best_honey.code, float(best_honey.fitness), (bee_end_time - bee_start_time))
            )
            # checkpoint.save_honey_model(state)
            flops = get_arch_flops(best_honey.code)
            logger.info('FLOPS Compress Rate: %.2f M/%.2f M(%.2f%%)' % (
                flops / 1000000, oriflops / 1000000, 100. * (oriflops - flops) / oriflops))

        else:
            best_honey.code = args.best_honey

        # Model
        print('==> Building model..')
        if args.arch == 'vgg_cifar':
            model = import_module(f'model.{args.arch}').BeeVGG(args.cfg, honeysource=best_honey.code,food_scale=args.food_scale).to(device)
        elif args.arch == 'resnet_cifar':
            model = import_module(f'model.{args.arch}').resnet_new(args.cfg, honey=best_honey.code).to(device)
        elif args.arch == 'googlenet':
            model = import_module(f'model.{args.arch}').googlenet(honey=best_honey.code).to(device)
        elif args.arch == 'densenet':
            model = import_module(f'model.{args.arch}').densenet(honey=best_honey.code).to(device)

        # if args.best_honey_s:
        #     bestckpt = torch.load(args.best_honey_s)
        #     model.load_state_dict(bestckpt)
        # else:
        #     if args.arch == 'vgg_cifar':
        #         get_vgg_honey_state(model, args.random_rule)
        #     elif args.arch == 'resnet_cifar':
        #         load_resnet_honey_model(model, args.random_rule)
        #     elif args.arch == 'googlenet':
        #         get_google_honey_model(model, args.random_rule)
        #     elif args.arch == 'densenet':
        #         get_dense_honey_model(model, args.random_rule)
        #
        if args.best_honey_s:
            bestckpt = torch.load(args.best_honey_s)
            model.load_state_dict(bestckpt)
        else:
            model.load_state_dict(best_honey_state)

        checkpoint.save_honey_model(model.state_dict())

        print(args.random_rule + ' Done!')

        if len(args.gpus) != 1:
            model = nn.DataParallel(model, device_ids=args.gpus)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        start_epoch = 0
        if args.best_honey == None:
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                  weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_step, gamma=0.1)
            code = best_honey.code
            start_epoch = 0

    else:
        # Model
        resumeckpt = torch.load(args.resume)
        state_dict = resumeckpt['state_dict']
        code = resumeckpt['honey_code']
        print('==> Building model..')
        if args.arch == 'vgg_cifar':
            model = import_module(f'model.{args.arch}').VGG_new(args.cfg, honeysource=code,
                                                                food_scale=args.food_scale).to(device)
        elif args.arch == 'resnet_cifar':
            model = import_module(f'model.{args.arch}').resnet(args.cfg, honey=code).to(device)
        elif args.arch == 'googlenet':
            model = import_module(f'model.{args.arch}').googlenet(honey=code).to(device)
        elif args.arch == 'densenet':
            model = import_module(f'model.{args.arch}').densenet(honey=code).to(device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_step, gamma=0.1)

        model.load_state_dict(state_dict)
        optimizer.load_state_dict(resumeckpt['optimizer'])
        # scheduler.load_state_dict(resumeckpt['scheduler'])
        start_epoch = resumeckpt['epoch']

        if len(args.gpus) != 1:
            model = nn.DataParallel(model, device_ids=args.gpus)

    if args.test_only:
        test(model, loader.testLoader)

    else:

        for epoch in range(start_epoch, args.num_epochs):
            if epoch in [args.num_epochs * 0.5, args.num_epochs * 0.75]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
            train(model, optimizer, loader.trainLoader, args, epoch)
            test_acc = test(model, loader.testLoader)

            is_best = best_acc < test_acc
            best_acc = max(best_acc, test_acc)

            model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()

            state = {
                'state_dict': model_state_dict,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'honey_code': code
            }
            checkpoint.save_model(state, epoch + 1, is_best)

        logger.info('Best accurary: {:.3f}'.format(float(best_acc)))


if __name__ == '__main__':
    main()
# nohup /home/chengyijie/.conda/envs/chen_p38/bin/python3.7 -u /home/chengyijie/pycharm/ABCPruner/bee_cifar.py >/dev/null 2>&1 &