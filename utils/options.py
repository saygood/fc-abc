import argparse
import ast
import os

'''
Args for the BeePruning:
Gpus 
Data: dataset data_path 
Log: job_dir reset
Vheckpoint: resume refine
Net: arch cfg
Train: num_epochs train_batch_size eval_batch_size momentum lr lr_decay_step weight_decay
'''

parser = argparse.ArgumentParser(description='Prune via BeePruning')

parser.add_argument(
    '--from_scratch',
    action='store_true',
    help='Train from scratch?'
)

parser.add_argument(
    '--bee_from_scratch',
    action='store_true',
    help='Beepruning from scratch?'
)

parser.add_argument(
    '--label_smooth',
    action='store_true',
    help='Use Lable smooth criterion?'
)

parser.add_argument(
    '--split_optimizer',
    action='store_true',
    help='Split the weight parameter that need weight decay?'
)

parser.add_argument(
    '--warm_up',
    action='store_true',
    help='Use warm up LR?'
)
parser.add_argument(
	'--gpus',
	type=int,
	nargs='+',
	default=[1],
	help='Select gpu_id to use. default:[0]',
)

parser.add_argument(
    '--sr',
    type=float,
    default=0.0001,
    help = 'sparsity factor for sparsity training',
)

parser.add_argument(
    '--s',
    type=bool,
    default=False,
    help = 'the sparsity factor for sparsity training',
)

parser.add_argument(
	'--data_set',
	type=str,
	default='cifar10',
	help='Select dataset to train. default:cifar10',
)


parser.add_argument(
	'--data_path',
	type=str,
	default='./data',
	help='The dictionary where the input is stored. default:',
)

parser.add_argument(
    '--job_dir',
    type=str,
    default='./experiments',
    help='The directory where the summaries will be stored. default:./experiments',
)

parser.add_argument(
    '--reset',
    action='store_true',
    help='reset the directory?'
)

parser.add_argument(
	'--resume',
	type=str,
	default=None,
	help='Load the model from the specified checkpoint.'
)

parser.add_argument(
	'--refine',
	type=str,
	default=None,
	help='Path to the model to be fine-tuned.'
)

## Training
parser.add_argument(
    '--arch',
    type=str,
    default='vgg_cifar',
    help='Architecture of model. default:vgg_cifar'
)

parser.add_argument(
    '--cfg',
    type=str,
    default='vgg16',
    help='Detail architecuture of model. default:vgg16'
)

parser.add_argument(
    '--num_epochs',
    type=int,
    default=150,
    help='The num of epochs to train. default:150'
)

parser.add_argument(
    '--train_batch_size',
    type=int,
    default=256,
    help='Batch size for training. default:256'
)

parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=256,
    help='Batch size for validation. default:256'
)

parser.add_argument(
    '--momentum',
    type=float,
    default=0.9,
    help='Momentum for MomentumOptimizer. default:0.9'
)

parser.add_argument(
    '--lr',
    type=float,
    default=0.01,
    help='Learning rate for train. default:0.1'
)

parser.add_argument(
    '--lr_decay_step',
    type=int,
    default=[50],
    help='the iterval of learn rate decay. default:30'
)

parser.add_argument(
    '--weight_decay',
    type=float,
    default=5e-3,
    help='The weight decay of loss. default:1e-4'
)

parser.add_argument(
    '--random_rule',
    type=str,
    default='l1_bn_pretrain',
    help='Weight initialization criterion after random clipping. default:default optional:default,random_pretrain,l1_pretrain'
)

parser.add_argument(
    '--test_only',
    action='store_true',
    help='Test only?')

#Beepruning
parser.add_argument(
    '--honey_model',
    type=str,
    default='./data/model_193.pt',
    help='Path to the model wait for Beepruning. default:None'
)

parser.add_argument(
    '--calfitness_epoch',
    type=int,
    default=2,
    help='Calculate fitness of honey source: training epochs. default:2'
)

parser.add_argument(
    '--max_cycle',
    type=int,
    default=2,
    help='Search for best pruning plan times. default:10'
)

parser.add_argument(
    '--max_preserve',
    type=int,
    default=9,
    help='Minimum percent of prune per layer'
)
parser.add_argument(
    '--pr',
    type=float,
    default=0.74,
    help='pruning ratio of model'
)

parser.add_argument(
    '--preserve_type',
    type = str,
    default = 'layerwise',
    help = 'The preserve ratio of each layer or the preserve ratio of the entire network'

)

parser.add_argument(
    '--food_number',
    type=int,
    default=3,
    help='Food number'
)

parser.add_argument(
    '--food_dimension',
    type=int,
    default=13,
    help='Food dimension: num of conv layers. default: vgg16->13 conv layer to be pruned'
)

parser.add_argument(
	'--food_scale',
	type=list,
	default=[],
	help='obtain the network structure. default:[]',
)

parser.add_argument(
    '--food_limit',
    type=int,
    default=5,
    help='Beyond this limit, the bee has not been renewed to become a scout bee,default:5'
)

parser.add_argument(
    '--honeychange_num',
    type=int,
    default=2,
    help='Number of codes that the nectar source changes each time'
)

parser.add_argument(
    '--best_honey',
    type=int,
    nargs='+',
    default=None,
    help='If this hyper-parameter exists, skip bee-pruning and fine-tune from this prune method'
)

parser.add_argument(
    '--best_honey_s',
    type=str,
    default=None,
    help='Path to the best_honey'
)

parser.add_argument(
    '--best_honey_past',
    type=int,
    nargs='+',
    default=None,
)

args = parser.parse_args()

netcfg = {
    'vgg16':13,
    'resnet18':8,
    'resnet56':27,
    'resnet110':54,
    'resnet34' : 16,
    'resnet50' : 16,
    'resnet101' : 33,
    'resnet152' : 50,
    'googlenet': 9,
    'densenet': 36,
}

args.food_dimension = netcfg[args.cfg]

if args.resume is not None and not os.path.isfile(args.resume):
    raise ValueError('No checkpoint found at {} to resume'.format(args.resume))

if args.refine is not None and not os.path.isfile(args.refine):
    raise ValueError('No checkpoint found at {} to refine'.format(args.refine))

