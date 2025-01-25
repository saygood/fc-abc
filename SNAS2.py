
import utils.common as utils
import copy
import math
import argparse
import sys
import ast
import numpy as np
import heapq
import random
import os
import time
from argparse import ArgumentParser
import torch
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data.distributed

from ssd.model import SNAS, Loss, ResNet_NAS
from ssd.resnet import ResNet50_new
from ssd.utils import dboxes300_coco, Encoder
from ssd.logger import Logger, BenchLogger
from ssd.evaluate import evaluate
from ssd.train import train_loop, tencent_trick, load_checkpoint, benchmark_train_loop, benchmark_inference_loop
from ssd.data import get_train_loader, get_val_loader, get_val_dataset, get_val_dataloader, get_coco_ground_truth
from math import log
from thop import profile

import dllogger as DLLogger

# Apex imports
try:
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    raise ImportError("Please install APEX from https://github.com/nvidia/apex")

parser = ArgumentParser(description="Train Single Shot MultiBox Detector" " on COCO")
parser.add_argument('--data', '-d', type=str, default='/2023110185/coco2017', help='path to test and training data files')
parser.add_argument('--epochs', '-e', type=int, default=65, help='number of epochs for training')
parser.add_argument('--batch-size', '--bs', type=int, default=128, help='number of examples for each iteration')
parser.add_argument('--eval-batch-size', '--ebs', type=int, default=32, help='number of examples for each evaluation iteration')
parser.add_argument('--no-cuda', action='store_true', help='use available GPUs')
parser.add_argument('--seed', '-s', type=int, help='manually set random seed for torch')
parser.add_argument('--checkpoint', type=str, default=None, help='path to model checkpoint file')
parser.add_argument('--torchvision-weights-version', type=str, default="IMAGENET1K_V2", choices=['IMAGENET1K_V1', 'IMAGENET1K_V2', 'DEFAULT'], help='The torchvision weights version to use when --checkpoint is not specified')
parser.add_argument('--save', type=str, default='/2023110185/2023110185/DeepLearningExamples-master/PyTorch/Detection/SSD/save_models', help='save model checkpoints in the specified directory')
parser.add_argument('--mode', type=str, default='training', choices=['training', 'evaluation', 'benchmark-training', 'benchmark-inference'])
parser.add_argument('--evaluation', nargs='*', type=int, default=[21, 31, 37, 42, 48, 53, 59, 64], help='epochs at which to evaluate')
parser.add_argument('--multistep', nargs='*', type=int, default=[43, 54], help='epochs at which to decay learning rate')
parser.add_argument('--job_dir',type=str,default='./experiments/Training/resnet50_imagenet/bee_search2',help='The directory where the summaries will be stored. default:./experiments',)

# Hyperparameters
parser.add_argument('--learning-rate', '--lr', type=float, default=2.6e-3, help='learning rate')
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='momentum argument for SGD optimizer')
parser.add_argument('--weight-decay', '--wd', type=float, default=0.0005, help='momentum argument for SGD optimizer') 
parser.add_argument('--warmup', type=int, default=None)
parser.add_argument('--benchmark-iterations', type=int, default=20, metavar='N', help='Run N iterations while benchmarking (ignored when training and validation)')
parser.add_argument('--benchmark-warmup', type=int, default=20, metavar='N', help='Number of warmup iterations for benchmarking')
parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
parser.add_argument('--backbone-path', type=str, default=None, help='Path to chekcpointed backbone. It should match the' ' backbone model declared with the --backbone argument.'
                            ' When it is not provided, pretrained model from torchvision'
                            ' will be downloaded.')
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument("--amp", dest='amp', action="store_true", help="Enable Automatic Mixed Precision (AMP).")
parser.add_argument("--no-amp", dest='amp', action="store_false", help="Disable Automatic Mixed Precision (AMP).")
parser.set_defaults(amp=True)
parser.add_argument("--allow-tf32", dest='allow_tf32', action="store_true", help="Allow TF32 computations on supported GPUs.")
parser.add_argument("--no-allow-tf32", dest='allow_tf32', action="store_false", help="Disable TF32 computations.")
parser.set_defaults(allow_tf32=True)
parser.add_argument('--data-layout', default="channels_last", choices=['channels_first', 'channels_last'], help="Model data layout. It's recommended to use channels_first with --no-amp")
parser.add_argument('--log-interval', type=int, default=20, help='Logging interval.')
parser.add_argument('--json-summary', type=str, default=None, help='If provided, the json summary will be written to' 'the specified file.')

# Distributed
parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK',0), type=int, help='Used for multi-process training. Can either be manually set ' +
                            'or automatically set by using \'python -m multiproc\'.')
#Beepruning
parser.add_argument('--honey_model',type=str,default='/home/chengyijie/pycharm/ABCPruner/experiments/Training/resnet50_imagenet/cnn/checkpoint/model_90.pt',help='Path to the model wait for Beepruning. default:None')
parser.add_argument('--calfitness_epoch',type=int,default=4,help='Calculate fitness of honey source: training epochs. default:2')
parser.add_argument('--max_cycle',type=int, default=5, help='Search for best pruning plan times. default:10')
parser.add_argument('--cell',type=int,default=3, help='Minimum percent of prune per layer')
parser.add_argument('--pr',type=float,default=0.495, help='pruning ratio of model')
parser.add_argument('--preserve_type',type = str,default = 'layerwise',help = 'The preserve ratio of each layer or the preserve ratio of the entire network')
parser.add_argument('--food_number',type=int,default=5, help='Food number')
parser.add_argument('--food_dimension',type=int,default=37, help='Food dimension: num of conv layers. default: vgg16->13 conv layer to be pruned')
parser.add_argument('--food_scale',type=list,default=[], help='obtain the network structure. default:[]',)
parser.add_argument('--food_limit',type=int,default=5, help='Beyond this limit, the bee has not been renewed to become a scout bee,default:5')
parser.add_argument('--honeychange_num',type=int,default=7, help='Number of codes that the nectar source changes each time')
parser.add_argument('--best_honey',type=int,nargs='+',default=None, help='If this hyper-parameter exists, skip bee-pruning and fine-tune from this prune method')
parser.add_argument('--best_honey_s',type=str,default=None, help='Path to the best_honey')
parser.add_argument( '--best_honey_past', type=int, nargs='+', default=None,)

args = parser.parse_args()
args.food_scale = [64, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 256, 512, 1024, 2048]
food_dimension = 37
args.no_cuda = False
args.N_gpu = 1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

device = torch.device(f"cuda:{0}") if torch.cuda.is_available() else 'cpu'
checkpoint = utils.checkpoint(args)
logger_nas = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
input = torch.randn(1, 3, 224, 224)
origin_model = ResNet50_new()
oriflops, oriparams = profile(nn.Sequential(*list(origin_model.children())[:6]), inputs=(input, ))
# print(self.feature_extractor)

#Our artificial bee colony code is based on the framework at https://www.cnblogs.com/ybl20000418/p/11366576.html 
#Define BeeGroup 
class BeeGroup():
    """docstring for BeeGroup"""
    def __init__(self):
        super(BeeGroup, self).__init__() 
        self.code = [] #size : num of conv layers  value:{1,2,3,4,5,6,7,8,9,10}
        self.fitness = 0
        self.rfitness = 0 
        self.trail = 0

#Initialize global element
best_honey = BeeGroup()
NectraSource = []
EmployedBee = []
OnLooker = []
best_honey_state = {}
vis_dict = []


def generate_mean_std(args):
    mean_val = [0.485, 0.456, 0.406]
    std_val = [0.229, 0.224, 0.225]

    mean = torch.tensor(mean_val).cuda()
    std = torch.tensor(std_val).cuda()

    view = [1, len(mean_val), 1, 1]

    mean = mean.view(*view)
    std = std.view(*view)

    return mean, std


def train(train_loop_func, logger, args):
    # Check that GPUs are actually available
    use_cuda = True

    # Setup multi-GPU if necessary
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.N_gpu = torch.distributed.get_world_size()
    else:
        args.N_gpu = 4

    if args.seed is None:
        args.seed = np.random.randint(1e4)

    if args.distributed:
        args.seed = (args.seed + torch.distributed.get_rank()) % 2**32
    print("Using seed = {}".format(args.seed))
    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)

    # Setup data, defaults
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    cocoGt = get_coco_ground_truth(args)

    train_loader = get_train_loader(args, args.seed - 2**31)

    val_dataset = get_val_dataset(args)
    val_dataloader = get_val_dataloader(val_dataset, args)

    ssd300 = SNAS(backbone=ResNet_NAS('resnet50_nas', args.best_honey)).to(device)
    # if args.N_gpu > 1:
    #     ssd300 = nn.DataParallel(ssd300, device_ids=args.gpus)
    args.learning_rate = args.learning_rate * args.N_gpu * (args.batch_size / 32)
    start_epoch = 0
    iteration = 0
    loss_func = Loss(dboxes)

    if use_cuda:
        loss_func.cuda()

    optimizer = torch.optim.SGD(tencent_trick(ssd300), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=args.multistep, gamma=0.1)

    if args.distributed:
        ssd300 = DDP(ssd300)

    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            load_checkpoint(ssd300.module if args.distributed else ssd300, args.checkpoint)
            checkpoint = torch.load(args.checkpoint,
                                    map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
            start_epoch = checkpoint['epoch']
            iteration = checkpoint['iteration']
            scheduler.load_state_dict(checkpoint['scheduler'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('Provided checkpoint is not path to a file')
            return

    inv_map = {v: k for k, v in val_dataset.label_map.items()}

    total_time = 0

    if args.mode == 'evaluation':
        acc = evaluate(ssd300, val_dataloader, cocoGt, encoder, inv_map, args)
        if args.local_rank == 0:
            print('Model precision {} mAP'.format(acc))
        return

    scaler = torch.amp.GradScaler(enabled=args.amp, device_type='cuda')
    mean, std = generate_mean_std(args)

    for epoch in range(start_epoch, args.epochs):
        start_epoch_time = time.time()
        iteration = train_loop_func(ssd300, loss_func, scaler,
                                    epoch, optimizer, train_loader, val_dataloader, encoder, iteration,
                                    logger, args, mean, std)
        if args.mode in ["training", "benchmark-training"]:
            scheduler.step()
        end_epoch_time = time.time() - start_epoch_time
        total_time += end_epoch_time

        if args.local_rank == 0:
            logger.update_epoch_time(epoch, end_epoch_time)

        if epoch in args.evaluation:
            acc = evaluate(ssd300, val_dataloader, cocoGt, encoder, inv_map, args)

            if args.local_rank == 0:
                logger.update_epoch(epoch, acc)

        if args.save and args.local_rank == 0:
            print("saving model...")
            obj = {'epoch': epoch + 1,
                   'iteration': iteration,
                   'optimizer': optimizer.state_dict(),
                   'scheduler': scheduler.state_dict(),
                   'label_map': val_dataset.label_info}
            if args.distributed:
                obj['model'] = ssd300.module.state_dict()
            else:
                obj['model'] = ssd300.state_dict()
            os.makedirs(args.save, exist_ok=True)
            save_path = os.path.join(args.save, f'epoch_{epoch}.pt')
            torch.save(obj, save_path)
            logger.log('model path', save_path)
        train_loader.reset()
    DLLogger.log((), { 'total time': total_time })
    logger.log_summary()


def log_params(logger, args):
    logger.log_params({
        "dataset path": args.data,
        "epochs": args.epochs,
        "batch size": args.batch_size,
        "eval batch size": args.eval_batch_size,
        "no cuda": args.no_cuda,
        "seed": args.seed,
        "checkpoint path": args.checkpoint,
        "mode": args.mode,
        "eval on epochs": args.evaluation,
        "lr decay epochs": args.multistep,
        "learning rate": args.learning_rate,
        "momentum": args.momentum,
        "weight decay": args.weight_decay,
        "lr warmup": args.warmup,
        "backbone": args.backbone,
        "backbone path": args.backbone_path,
        "num workers": args.num_workers,
        "AMP": args.amp,
        "precision": 'amp' if args.amp else 'fp32',
    })


def load_resnet_honey_model(model, random_rule):

    cfg = {'resnet18': [2,2,2,2],
           'resnet34': [3,4,6,3],
           'resnet50': [3,4,6,3],
           'resnet101': [3,4,23,3],
           'resnet152': [3,8,36,3],}

    global oristate_dict
    state_dict = model.state_dict()
        
    current_cfg = cfg[args.cfg]
    last_select_index = None
    all_honey_conv_name = []
    all_honey_bn_name = []
    all_honey_conv_weight = []
    # for name in state_dict.keys():
    #     print(name)
    conv_name = 'conv1'
    conv_weight_name = conv_name + '.weight'
    all_honey_conv_name.append(conv_name)
    all_honey_bn_name.append('bn1')
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

        for index_i, i in enumerate(select_index):
            state_dict[conv_weight_name][index_i] = oristate_dict[conv_weight_name][i]
            state_dict['bn1.weight'][index_i] = oristate_dict['bn1.weight'][i]
            state_dict['bn1.bias'][index_i] = oristate_dict['bn1.bias'][i]
            state_dict['bn1.running_var'][index_i] = oristate_dict['bn1.running_var'][i]
            state_dict['bn1.running_mean'][index_i] = oristate_dict['bn1.running_mean'][i]

        last_select_index = select_index
        # logger_nas.info('last_select_index{}'.format(last_select_index))

    else:
        state_dict[conv_weight_name] = oriweight
        state_dict['bn1.weight'] = oristate_dict['bn1.weight']
        state_dict['bn1.bias'] = oristate_dict['bn1.bias']
        state_dict['bn1.running_var'] = oristate_dict['bn1.running_var']
        state_dict['bn1.running_mean'] = oristate_dict['bn1.running_mean']
        last_select_index = None

    for layer, num in enumerate(current_cfg): # 3 4 6 3
        layer_name = 'layer' + str(layer + 1) + '.'
        if args.cfg == 'resnet18' or args.cfg == 'resnet34':
            iter = 2  # the number of convolution layers in a block, except for shortcut
        else:
            iter = 3
        Flag = True
        for k in range(num): #
            last_select_index_temp = last_select_index
            for l in range(iter):
                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                bn_name = layer_name + str(k) + '.bn' + str(l + 1)
                conv_weight_name = conv_name + '.weight'
                bn_weight_name = bn_name + '.weight'
                all_honey_conv_name.append(conv_name)
                all_honey_bn_name.append(bn_name)
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
                    if last_select_index_temp is not None:
                        # print('1x1')
                        # print(state_dict[conv_weight_name].shape)
                        # print(oristate_dict[conv_weight_name].shape)
                        # print(select_index)
                        # print(last_select_index_temp)
                        # print('1x1end')
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index_temp):
                                state_dict[conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]
                    last_select_index_temp = select_index
                    for index_i, i in enumerate(select_index):
                        state_dict[bn_weight_name][index_i] = oristate_dict[bn_weight_name][i]
                        state_dict[bn_name + '.bias'][index_i] = oristate_dict[bn_name + '.bias'][i]
                        state_dict[bn_name + '.running_var'][index_i] = oristate_dict[bn_name + '.running_var'][i]
                        state_dict[bn_name + '.running_mean'][index_i] = oristate_dict[bn_name + '.running_mean'][i]
                else:
                    if last_select_index_temp is not None:
                        # print('1x1')
                        # print(state_dict[conv_name].shape)
                        # print(oristate_dict[conv_name].shape)
                        # print(last_select_index_1x1)
                        # print('1x1end')
                        select_index = [i for i in range(currentfilter_num)]
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index_temp):
                                state_dict[conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        state_dict[conv_weight_name] = oriweight
                    state_dict[bn_weight_name] = oristate_dict[bn_weight_name]
                    state_dict[bn_name + '.bias'] = oristate_dict[bn_name + '.bias']
                    state_dict[bn_name + '.running_var'] = oristate_dict[bn_name + '.running_var']
                    state_dict[bn_name + '.running_mean'] = oristate_dict[bn_name + '.running_mean']

                    last_select_index_temp = None
            if layer_name + '0.downsample.0.weight' in oristate_dict.keys() and Flag: #layer4.0.downsample.0.weight
                Flag = False
                conv_name = layer_name + '0.downsample.0'
                conv_weight_name = conv_name + '.weight'
                bn_name = layer_name + '0.downsample.1'
                # print(conv_weight_name)
                all_honey_conv_name.append(conv_name)
                all_honey_bn_name.append(bn_name)
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

                    if last_select_index is not None:
                        # print('1x1')
                        # print(state_dict[conv_name].shape)
                        # print(oristate_dict[conv_name].shape)
                        # print(select_index)
                        # print(last_select_index_1x1)
                        # print('1x1end')
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]

                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]
                    for index_i, i in enumerate(select_index):
                        state_dict[bn_name + '.weight'][index_i] = oristate_dict[bn_name + '.weight'][i]
                        state_dict[bn_name + '.bias'][index_i] = oristate_dict[bn_name + '.bias'][i]
                        state_dict[bn_name + '.running_var'][index_i] = oristate_dict[bn_name + '.running_var'][i]
                        state_dict[bn_name + '.running_mean'][index_i] = oristate_dict[bn_name + '.running_mean'][i]
                else:
                    if last_select_index is not None:
                        # print('1x1')
                        # print(state_dict[conv_name].shape)
                        # print(oristate_dict[conv_name].shape)
                        # print(last_select_index_1x1)
                        # print('1x1end')
                        select_index = [i for i in range(currentfilter_num)]
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        state_dict[conv_weight_name] = oriweight
                    state_dict[bn_name + '.weight'][index_i] = oristate_dict[bn_name + '.weight'][i]
                    state_dict[bn_name + '.bias'][index_i] = oristate_dict[bn_name + '.bias'][i]
                    state_dict[bn_name + '.running_var'][index_i] = oristate_dict[bn_name + '.running_var'][i]
                    state_dict[bn_name + '.running_mean'][index_i] = oristate_dict[bn_name + '.running_mean'][i]
            last_select_index = last_select_index_temp
    conv_name = 'fc'
    conv_weight_name = conv_name + '.weight'
    all_honey_conv_name.append(conv_name)
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
            if name not in all_honey_conv_name:
                state_dict[name + '.weight'] = oristate_dict[name + '.weight']

        elif isinstance(module, nn.Linear):
            if name not in all_honey_conv_name:
                state_dict[name + '.weight'] = oristate_dict[name + '.weight']
                state_dict[name + '.bias'] = oristate_dict[name + '.bias']
                state_dict[name + '.running_mean'] = oristate_dict[name + '.running_mean']
                state_dict[name + '.running_var'] = oristate_dict[name + '.running_var']

    #for param_tensor in state_dict:
        #logger_nas.info('param_tensor {}\tType {}\n'.format(param_tensor,state_dict[param_tensor].size()))
    #for param_tensor in model.state_dict():
        #logger_nas.info('param_tensor {}\tType {}\n'.format(param_tensor,model.state_dict()[param_tensor].size()))
 

    model.load_state_dict(state_dict)


def change_depth(honey, honeychange_num):
    param2change = np.random.randint(0, food_dimension - 2, honeychange_num)
    for j in range(honeychange_num):
        if honey[param2change[j]] - 1 > 0:
            honey[param2change[j]] -= 1
        else:
            honey[param2change[j]] = 1
    return honey


def decode(honey):
    # print("honey:",honey)
    food_scale = [64, 64, 64, 256, 64, 64, 256, 64, 64, 256, 128, 128, 512, 128, 128, 512, 128, 128, 512, 128, 128,
                  512, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256,
                  1024, 512, 512, 2048, 512, 512, 2048, 512, 512, 2048]
    t1 = [3,4,6,3]
    s1 = honey[1:33]
    s2 = honey[33:37]
    o1 = [0] * 48
    index1 = 0
    index2 = 0
    index3 = 0
    for i in range(48):
        if (i+1) % 3 != 0:
            o1[i] = s1[index1]
            index1 += 1
        else:
            index3 += 1
            o1[i] = s2[index2]
            if index3 == t1[index2]:
                index3 = 0
                index2 += 1

    honey_new = [honey[0]] + o1
    struc = []
    for i in range(len(food_scale)):
        struc.append(int(food_scale[i] * honey_new[i] / (10 * 2 ** int(log(food_scale[i], 2) - log(food_scale[0], 2)))))
    return struc


# Calculate fitness of a honey source
def calculationFitness(honey, args):
    global best_honey
    global best_honey_state
    global vis_dict

    vis_dict.append(honey)
    args.distributed = False
    if args.seed is None:
        args.seed = np.random.randint(1e4)
    # Setup data, defaults
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    cocoGt = get_coco_ground_truth(args)
    #dataset loading
    train_loader = get_train_loader(args, args.seed - 2**31)
    # val_loader = get_val_loader(args, args.seed - 2**31)
    # val_dataset = get_val_dataset(args)
    # val_dataloader = get_val_dataloader(val_dataset, args)
    #model    
    # ssd300 = SNAS(backbone=ResNet_NAS(backbone=args.backbone, backbone_path=args.backbone_path, weights=args.torchvision_weights_version))
    ssd300 = SNAS(backbone=ResNet_NAS('resnet50_nas', honey=honey)).to(device)
    args.learning_rate = args.learning_rate * args.N_gpu * (args.batch_size / 32)
    start_epoch = 0
    iteration = 0
    loss_func = Loss(dboxes)
    loss_func.cuda()

    optimizer = torch.optim.SGD(tencent_trick(ssd300), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=args.multistep, gamma=0.1)
    fit_accurary = utils.AverageMeter()

    total_time = 0
    scaler = torch.amp.GradScaler(enabled=args.amp)
    mean, std = generate_mean_std(args)
    
    for epoch in range(start_epoch, args.calfitness_epoch):
        ######test#####
        # if epoch>1:
        #     for nbatch, data in enumerate(val_loader):
        #         img = data[0][0][0]
        #         bbox = data[0][1][0]
        #         label = data[0][2][0]
        #         label = label.type(torch.cuda.LongTensor)
        #         bbox_offsets = data[0][3][0]
        #         bbox_offsets = bbox_offsets.cuda()
        #         img.sub_(mean).div_(std)
        #         if not args.no_cuda:
        #             img = img.cuda()
        #             bbox = bbox.cuda()
        #             label = label.cuda()
        #             bbox_offsets = bbox_offsets.cuda()

        #         N = img.shape[0]
        #         if bbox_offsets[-1].item() == 0:
        #             print("No labels in batch")
        #             continue

        #         # output is ([N*8732, 4], [N*8732], need [N, 8732, 4], [N, 8732] respectively
        #         M = bbox.shape[0] // N
        #         bbox = bbox.view(N, M, 4)
        #         label = label.view(N, M)
                                        
        #         with torch.amp.autocast(enabled=args.amp, device_type='cuda'):
        #             if args.data_layout == 'channels_last':
        #                 img = img.to(memory_format=torch.channels_last)
        #             ploc, plabel = ssd300(img)

        #             ploc, plabel = ploc.float(), plabel.float()
        #             trans_bbox = bbox.transpose(1, 2).contiguous().cuda()
        #             gloc = Variable(trans_bbox, requires_grad=False)
        #             glabel = Variable(label, requires_grad=False)

        #             loss = loss_func(ploc, plabel, gloc, glabel)
        #             # print("--1--", loss.item())
                    
        #             if torch.isnan(loss):
        #                 fit_accurary.update(0.0, 1)
        #             else:
        #                 temp_acc = 0.0
        #                 temp = loss.item()
        #                 temp_acc = 0.5 - temp/100
        #                 fit_accurary.update(temp_acc, 1)
                
        ######test#####

        start_epoch_time = time.time()
        for nbatch, data in enumerate(train_loader):
            img = data[0][0][0]
            bbox = data[0][1][0]
            label = data[0][2][0]
            label = label.type(torch.cuda.LongTensor)
            bbox_offsets = data[0][3][0]
            bbox_offsets = bbox_offsets.cuda()
            img.sub_(mean).div_(std)
            if not args.no_cuda:
                img = img.cuda()
                bbox = bbox.cuda()
                label = label.cuda()
                bbox_offsets = bbox_offsets.cuda()

            N = img.shape[0]
            if bbox_offsets[-1].item() == 0:
                print("No labels in batch")
                continue

            # output is ([N*8732, 4], [N*8732], need [N, 8732, 4], [N, 8732] respectively
            M = bbox.shape[0] // N
            bbox = bbox.view(N, M, 4)
            label = label.view(N, M)
                                    
            with torch.amp.autocast(enabled=args.amp, device_type='cuda'):
                if args.data_layout == 'channels_last':
                    img = img.to(memory_format=torch.channels_last)
                ploc, plabel = ssd300(img)

                ploc, plabel = ploc.float(), plabel.float()
                trans_bbox = bbox.transpose(1, 2).contiguous().cuda()
                gloc = Variable(trans_bbox, requires_grad=False)
                glabel = Variable(label, requires_grad=False)

                loss = loss_func(ploc, plabel, gloc, glabel)
                # print("--1--", loss.item())
                if epoch>0:
                    if torch.isnan(loss):
                        fit_accurary.update(0.0, 1)
                    else:
                        temp_acc = 0.0
                        
                        temp = loss.item()
                        if temp > 100:
                            temp_acc = 0.0
                        else:
                            temp_acc = 1 - temp/100
                        fit_accurary.update(temp_acc/2, 1)

            if args.warmup is not None:
                warmup(optimizer, args.warmup, iteration, args.learning_rate)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            iteration += 1

        scheduler.step()
        end_epoch_time = time.time() - start_epoch_time
        total_time += end_epoch_time           

        train_loader.reset()
        # val_loader.reset()

    '''
    logger_nas.info('Honey Source fintness {:.2f}%\t\tTime {:.2f}s\n'.format(float(accurary.avg), (current_time - start_time)))
    '''
    if fit_accurary.avg == 0:
        fit_accurary.avg = 0.000001

    if fit_accurary.avg > best_honey.fitness:
        # best_honey_state = copy.deepcopy(model.module.state_dict() if len(args.gpus) > 1 else model.state_dict())
        best_honey.code = copy.deepcopy(honey)
        best_honey.fitness = fit_accurary.avg

    return fit_accurary.avg


def get_arch_flops(honey):

    backbone = ResNet50_new(honey=honey)
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(nn.Sequential(*list(backbone.children())[:6]), inputs=(input,))
    return flops


def is_legal_temp(honey):
    global vis_dict
    if honey in vis_dict:
        return False

    flops = get_arch_flops(honey)
    if flops / oriflops > (1 - args.pr) or flops / oriflops < (1 - args.pr - 0.01):
        # print('flops limit exceed')
        return False
    return True


def change_depth_temp(honey, honeychange_num):
    param2change = np.random.randint(0, food_dimension - 2, honeychange_num)
    for j in range(honeychange_num):
        if honey[param2change[j]] + 1 < int(args.food_scale[param2change[j]] / args.cell):
            honey[param2change[j]] += 1
        else:
            honey[param2change[j]] = int(args.food_scale[param2change[j]] / args.cell)
    return honey


#Initialize Bee-Pruning
def initialize():
    print('==> Initializing Honey_model..')
    global best_honey, NectraSource, EmployedBee, OnLooker

    honey_init = 0
    while honey_init < args.food_number:  # default:10
        NectraSource.append(copy.deepcopy(BeeGroup()))
        EmployedBee.append(copy.deepcopy(BeeGroup()))
        OnLooker.append(copy.deepcopy(BeeGroup()))
        honey = [random.randint(1, int(args.food_scale[j] / args.cell)) for j in range(37)]
        # honey[33:] = [random.randint(int(1 + 4 * (10 + 9) * j), int(args.max_preserve * args.food_scale[j + 33] / 64)) for j in range(4)]
        # honey[33:] = [random.randint(int(args.max_preserve * args.food_scale[j + 33] / 64), int(args.max_preserve * args.food_scale[j + 33] / 64)) for j in range(4)]
        honey[33] = random.randint(41, int(args.food_scale[33] / args.cell))
        honey[34] = random.randint(81, int(args.food_scale[34] / args.cell))
        honey[35] = int(args.food_scale[35] / args.cell)
        honey[36] = int(args.food_scale[36] / args.cell)
        honeychange_num = args.honeychange_num
        while True:
            if not is_legal_temp(honey):
                flops = get_arch_flops(honey)
                # print(flops / oriflops)
                if flops / oriflops < 1-args.pr-0.01:
                    change_depth_temp(honey, honeychange_num)
                else:
                    change_depth(honey, honeychange_num)
                honeychange_num -= 1
                if honeychange_num < 1:
                    honeychange_num = int(args.honeychange_num / 2)
                continue
            else:
                NectraSource[honey_init].code = copy.deepcopy(honey)
                break
        logger_nas.info(NectraSource[honey_init].code)

        #initialize honey souce
        NectraSource[honey_init].fitness = calculationFitness(NectraSource[honey_init].code, args)
        # ####优化初始搜索集合
        if NectraSource[honey_init].fitness < 0.3:
            continue
        # ####
        logger_nas.info(NectraSource[honey_init].fitness)
        NectraSource[honey_init].rfitness = 0
        NectraSource[honey_init].trail = 0

        #initialize employed bee  
        EmployedBee[honey_init].code = copy.deepcopy(NectraSource[honey_init].code)
        EmployedBee[honey_init].fitness=NectraSource[honey_init].fitness
        EmployedBee[honey_init].rfitness=NectraSource[honey_init].rfitness
        EmployedBee[honey_init].trail=NectraSource[honey_init].trail

        #initialize onlooker 
        OnLooker[honey_init].code = copy.deepcopy(NectraSource[honey_init].code)
        OnLooker[honey_init].fitness=NectraSource[honey_init].fitness
        OnLooker[honey_init].rfitness=NectraSource[honey_init].rfitness
        OnLooker[honey_init].trail=NectraSource[honey_init].trail
        honey_init += 1

    #initialize best honey
    best_honey.code = copy.deepcopy(NectraSource[0].code)
    best_honey.fitness = NectraSource[0].fitness
    best_honey.rfitness = NectraSource[0].rfitness
    best_honey.trail = NectraSource[0].trail

#Send employed bees to find better honey source
def sendEmployedBees():
    global NectraSource, EmployedBee
    for i in range(args.food_number):
        
        while 1:
            k = random.randint(0, args.food_number-1)
            if k != i:
                break

        EmployedBee[i].code = copy.deepcopy(NectraSource[i].code)
        Flag = True
        honeychange_num = args.honeychange_num
        while Flag:  # 思考一下怎么解决无限循环的问题

            param2change = np.random.randint(0, food_dimension - 1, honeychange_num)
            if is_legal(EmployedBee[i].code):
                R = np.random.uniform(-0.5, 0.5, honeychange_num)
            else:
                R = np.random.uniform(-0.5, 0, honeychange_num)
            for j in range(honeychange_num):
                EmployedBee[i].code[param2change[j]] = int(NectraSource[i].code[param2change[j]] + R[j] * (NectraSource[i].code[param2change[j]] - NectraSource[k].code[param2change[j]]))
                if EmployedBee[i].code[param2change[j]] < 1:
                    EmployedBee[i].code[param2change[j]] = 1
                if EmployedBee[i].code[param2change[j]] > args.food_scale[param2change[j]] / args.cell:
                    EmployedBee[i].code[param2change[j]] = int(args.food_scale[param2change[j]] / args.cell)
            if is_legal(EmployedBee[i].code):
                break
            else:
                honeychange_num -= 2
                if honeychange_num <= 0:
                    honeychange_num = 1
        EmployedBee[i].fitness = calculationFitness(EmployedBee[i].code, args)


        if EmployedBee[i].fitness > NectraSource[i].fitness:                
            NectraSource[i].code = copy.deepcopy(EmployedBee[i].code)              
            NectraSource[i].trail = 0  
            NectraSource[i].fitness = EmployedBee[i].fitness 
            
        else:          
            NectraSource[i].trail = NectraSource[i].trail + 1

#Calculate whether a Onlooker to update a honey source
def calculateProbabilities():
    global NectraSource
    
    maxfit = NectraSource[0].fitness

    for i in range(1, args.food_number):
        if NectraSource[i].fitness > maxfit:
            maxfit = NectraSource[i].fitness

    for i in range(args.food_number):
        NectraSource[i].rfitness = (0.9 * (NectraSource[i].fitness / maxfit)) + 0.1


def is_legal(honey):
    global vis_dict
    if honey in vis_dict:
        return False

    flops = get_arch_flops(honey)
    if flops / oriflops > (1-args.pr):
        # print('flops limit exceed')
        return False
    return True


#Send Onlooker bees to find better honey source
def sendOnlookerBees():
    global NectraSource, EmployedBee, OnLooker
    i = 0
    t = 0
    while t < args.food_number:
        R_choosed = random.uniform(0,1)
        if(R_choosed < NectraSource[i].rfitness):
            t += 1
            while 1:
                k = random.randint(0, args.food_number-1)
                if k != i:
                    break
            OnLooker[i].code = copy.deepcopy(NectraSource[i].code)
            honeychange_num = args.honeychange_num
            while True:
                param2change = np.random.randint(0, food_dimension-1, honeychange_num)
                if is_legal(OnLooker[i].code):
                    R = np.random.uniform(-0.5, 0.5, honeychange_num)
                else:
                    R = np.random.uniform(-0.5, 0, honeychange_num)

                for j in range(honeychange_num):
                    OnLooker[i].code[param2change[j]] = int(NectraSource[i].code[param2change[j]]+ R[j]*(NectraSource[i].code[param2change[j]]-NectraSource[k].code[param2change[j]]))
                    if OnLooker[i].code[param2change[j]] < 1:
                        OnLooker[i].code[param2change[j]] = 1
                    if OnLooker[i].code[param2change[j]] > args.food_scale[param2change[j]] / args.cell:
                        OnLooker[i].code[param2change[j]] = int(args.food_scale[param2change[j]] / args.cell)
                if is_legal(OnLooker[i].code):
                    break
                else:
                    honeychange_num -= 2
                    if honeychange_num <= 0:
                        honeychange_num = 1

            OnLooker[i].fitness = calculationFitness(OnLooker[i].code, args)

            if OnLooker[i].fitness > NectraSource[i].fitness:                
                NectraSource[i].code = copy.deepcopy(OnLooker[i].code)              
                NectraSource[i].trail = 0  
                NectraSource[i].fitness = OnLooker[i].fitness 
            else:          
                NectraSource[i].trail = NectraSource[i].trail + 1
        i += 1
        if i == args.food_number:
            i = 0

#If a honey source has not been update for args.food_limiet times, send a scout bee to regenerate it
def sendScoutBees():
    global  NectraSource, EmployedBee, OnLooker
    maxtrailindex = 0
    for i in range(args.food_number):
        if NectraSource[i].trail > NectraSource[maxtrailindex].trail:
            maxtrailindex = i
    if NectraSource[maxtrailindex].trail >= args.food_limit:
        honey = [random.randint(1, int(args.food_scale[j] / args.cell)) for j in range(37)]
        honey[36] = int(args.food_scale[36] / args.cell)
        honeychange_num = args.honeychange_num
        while True:
            if not is_legal(honey):
                change_depth(honey, honeychange_num)
                honeychange_num -= 1
                if honeychange_num < 1:
                    honeychange_num = int(args.honeychange_num / 2)
                continue
            else:
                # 初始化一个修剪结构
                NectraSource[maxtrailindex].code = copy.deepcopy(honey)
                break
        NectraSource[maxtrailindex].trail = 0
        NectraSource[maxtrailindex].fitness = calculationFitness(NectraSource[maxtrailindex].code, args)

 #Memorize best honey source
def memorizeBestSource():
    global best_honey, NectraSource
    for i in range(args.food_number):
        if NectraSource[i].fitness > best_honey.fitness:
            #print(NectraSource[i].fitness, NectraSource[i].code)
            #print(best_honey.fitness, best_honey.code)
            best_honey.code = copy.deepcopy(NectraSource[i].code)
            best_honey.fitness = NectraSource[i].fitness


def main():
    start_epoch = 0
    best_acc = 0.0
    best_acc_top1 = 0.0
    code = []

    if False:

        print('==> Building Model..')
        if args.arch == 'vgg':
            model = import_module(f'model.{args.arch}').VGG(num_classes=1000).to(device)
        elif args.arch == 'resnet':
            model = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)

        if len(args.gpus) != 1:
            model = nn.DataParallel(model, device_ids=[0,1,2,3])
 
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_step, gamma=0.1)

        if args.resume:
            print('=> Resuming from ckpt {}'.format(args.resume))
            ckpt = torch.load(args.resume, map_location=device)
            best_acc = ckpt['best_acc']
            start_epoch = ckpt['epoch']

            model.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            #scheduler.load_state_dict(ckpt['scheduler'])
            print('=> Continue from epoch {}...'.format(start_epoch))

    else:

        if True:
            if args.best_honey == None:

                start_time = time.time()
                
                bee_start_time = time.time()
                
                print('==> Start BeePruning..')

                initialize()

                #memorizeBestSource()

                for cycle in range(args.max_cycle):

                    current_time = time.time()
                    logger_nas.info('Search Cycle [{}]\t Best Honey Source {}\tBest Honey Source fitness {:.2f}%\tTime {:.2f}s\n'.format(cycle, best_honey.code, float(best_honey.fitness), (current_time - start_time)))
                    start_time = time.time()

                    sendEmployedBees() 
                      
                    calculateProbabilities()
                      
                    sendOnlookerBees()  
                      
                    #memorizeBestSource() 
                      
                    sendScoutBees() 
                      
                    #memorizeBestSource() 

                print('==> BeePruning Complete!')
                
                bee_end_time = time.time()
                logger_nas.info(
                    'Best Honey Source {}\tBest Honey Source fitness {:.2f}%\tTime Used{:.2f}s\n'
                    .format(best_honey.code, float(best_honey.fitness), (bee_end_time - bee_start_time))
                )

                flops = get_arch_flops(best_honey.code)
                args.best_honey = best_honey.code
                logger_nas.info('FLOPS Compress Rate: %.2f M/%.2f M(%.2f%%)' % (flops / 1000000, oriflops / 1000000, 100. * (oriflops - flops) / oriflops))
                #checkpoint.save_honey_model(state)
            else:
                best_honey.code = args.best_honey
                flops = get_arch_flops(best_honey.code)
                logger_nas.info('FLOPS Compress Rate: %.2f M/%.2f M(%.2f%%)' % (flops / 1000000, oriflops / 1000000, 100. * (oriflops - flops) / oriflops))
                #best_honey_state = torch.load(args.best_honey_s)

            # Modelmodel = import_module(f'model.{args.arch}').BeeVGG(honeysource=honey, num_classes=1000).to(device)
            print('==> Building model..')
            print(args.random_rule + ' Done!')

        else:
             # Model
            resumeckpt = torch.load(args.resume)
            state_dict = resumeckpt['state_dict']
            if args.best_honey_past == None:
                code = resumeckpt['honey_code']
            else:
                code = args.best_honey_past
            print('==> Building model..')
            if args.arch == 'vgg':
                model = import_module(f'model.{args.arch}').BeeVGG(honeysource=code, num_classes = 1000).to(device)
            elif args.arch == 'resnet':
                model = import_module(f'model.{args.arch}').ResNet50_new(honey=code).to(device)
            elif args.arch == 'googlenet':
                pass
            elif args.arch == 'densenet':
                pass

            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_step, gamma=0.1)

            model.load_state_dict(state_dict)
            optimizer.load_state_dict(resumeckpt['optimizer'])
            #scheduler.load_state_dict(resumeckpt['scheduler'])
            start_epoch = resumeckpt['epoch']

            if len(args.gpus) != 1:
                model = nn.DataParallel(model, device_ids=args.gpus)


    if args.test_only:
        test(model, testLoader,topk=(1, 5))
        
    else:

        args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
        if args.local_rank == 0:
            os.makedirs('./models', exist_ok=True)

        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.benchmark = True

        # write json only on the main thread
        args.json_summary = args.json_summary if args.local_rank == 0 else None

        if args.mode == 'benchmark-training':
            train_loop_func = benchmark_train_loop
            logger = BenchLogger('Training benchmark', log_interval=args.log_interval, json_output=args.json_summary)
            args.epochs = 1
        elif args.mode == 'benchmark-inference':
            train_loop_func = benchmark_inference_loop
            logger = BenchLogger('Inference benchmark', log_interval=args.log_interval, json_output=args.json_summary)
            args.epochs = 1
        else:
            train_loop_func = train_loop
            logger = Logger('Training logger', log_interval=args.log_interval, json_output=args.json_summary)

        log_params(logger, args)

        train(train_loop_func, logger, args)

if __name__ == '__main__':
    main()
