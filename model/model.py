# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import ssd.resnet
import time
import heapq

from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from ssd.resnet import ResNet50_new
from thop import profile

honey_new =[64, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256,
                       256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 256, 512, 1024, 2048]
# device = torch.device(f"cuda:0") if torch.cuda.is_available() else 'cpu'
# origin_model = ResNet50_new()
# ckpt = torch.load('/2023110185/2023110185/model_90.pt')

# origin_model.load_state_dict(ckpt['state_dict'])
# oristate_dict = origin_model.state_dict()

def load_resnet_honey_model(model, random_rule):

    cfg = {'resnet18': [2,2,2,2],
           'resnet34': [3,4,6,3],
           'resnet50': [3,4,6,3],
           'resnet101': [3,4,23,3],
           'resnet152': [3,8,36,3],}

    global oristate_dict
    state_dict = model.state_dict()
        
    current_cfg = cfg['resnet50']
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
        if current_cfg == 'resnet18' or current_cfg == 'resnet34':
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


class ResNet_NAS(nn.Module):
    def __init__(self, backbone='resnet50_nas', backbone_path=None, honey=None, flag= False, weights="IMAGENET1K_V1"):
        super().__init__()
        if honey is None:
            honey = honey_new
        if flag:
            honey = None
        if  backbone == 'resnet50_nas':
            backbone = ResNet50_new(honey=honey)
            # t1 = time.time()
            # load_resnet_honey_model(backbone, 'l1_pretrain')
            # t2 = time.time()
            # print(f"loading time{t2-t1:.6f}")
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        else:
            exit(0)
        if backbone_path:
            backbone.load_state_dict(torch.load(backbone_path))

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:6])
        # print(self.feature_extractor)

        conv4_block1 = self.feature_extractor[-1][0]

        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class ResNet(nn.Module):
    def __init__(self, backbone='resnet50', backbone_path=None, honey=None, weights="IMAGENET1K_V1"):
        super().__init__()
        if backbone == 'resnet18':
            backbone = resnet18(weights=None if backbone_path else weights)
            self.out_channels = [256, 512, 512, 256, 256, 128]
        elif backbone == 'resnet34':
            backbone = resnet34(weights=None if backbone_path else weights)
            self.out_channels = [256, 512, 512, 256, 256, 256]
        elif backbone == 'resnet50_nas':
            backbone = resnet50_nas(weights=None if backbone_path else weights)
            ###????###
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        elif backbone == 'resnet50':
            backbone = resnet50(weights=None if backbone_path else weights)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        elif backbone == 'resnet101':
            backbone = resnet101(weights=None if backbone_path else weights)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        else:  # backbone == 'resnet152':
            backbone = resnet152(weights=None if backbone_path else weights)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        if backbone_path:
            backbone.load_state_dict(torch.load(backbone_path))

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]

        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


# class SSD300(nn.Module):
#     def __init__(self, backbone=ResNet('resnet50')):
#         super().__init__()

#         self.feature_extractor = backbone
#         # print(backbone)
#         self.label_num = 81  # number of COCO classes
#         self._build_additional_features(self.feature_extractor.out_channels)
#         self.num_defaults = [4, 6, 6, 6, 4, 4]
#         self.loc = []
#         self.conf = []

#         for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
#             self.loc.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
#             self.conf.append(nn.Conv2d(oc, nd * self.label_num, kernel_size=3, padding=1))

#         self.loc = nn.ModuleList(self.loc)
#         self.conf = nn.ModuleList(self.conf)
#         self._init_weights()

#     def _build_additional_features(self, input_size):
#         # [1024, 512, 512, 256, 256, 256]
#         self.additional_blocks = []
#         #[1024, 512, 512, 256, 256 ] 
#         #[256, 256, 128, 128, 128]
#         #[512, 512, 256, 256, 256] 
#         for i, (input_size, output_size, channels) in enumerate(zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128])):
#             if i < 3:
#                 layer = nn.Sequential(
#                     nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
#                     nn.BatchNorm2d(channels),
#                     nn.ReLU(inplace=True),
#                     nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False),
#                     nn.BatchNorm2d(output_size),
#                     nn.ReLU(inplace=True),
#                 )
#             else:
#                 layer = nn.Sequential(
#                     nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
#                     nn.BatchNorm2d(channels),
#                     nn.ReLU(inplace=True),
#                     nn.Conv2d(channels, output_size, kernel_size=3, bias=False),
#                     nn.BatchNorm2d(output_size),
#                     nn.ReLU(inplace=True),
#                 )

#             self.additional_blocks.append(layer)

#         self.additional_blocks = nn.ModuleList(self.additional_blocks)

#     def _init_weights(self):
#         layers = [*self.additional_blocks, *self.loc, *self.conf]
#         for layer in layers:
#             for param in layer.parameters():
#                 if param.dim() > 1: nn.init.xavier_uniform_(param)

#     # Shape the classifier to the view of bboxes
#     def bbox_view(self, src, loc, conf):
#         ret = []
#         for s, l, c in zip(src, loc, conf):
#             ret.append((l(s).reshape(s.size(0), 4, -1), c(s).reshape(s.size(0), self.label_num, -1)))

#         locs, confs = list(zip(*ret))
#         locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
#         return locs, confs

#     def forward(self, x):
#         x = self.feature_extractor(x)

#         detection_feed = [x]
#         for l in self.additional_blocks:
#             x = l(x)
#             detection_feed.append(x)

#         # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
#         locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)

#         # For SSD 300, shall return nbatch x 8732 x {nlabels, nlocs} results
#         return locs, confs


class Loss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """
    def __init__(self, dboxes):
        super(Loss, self).__init__()
        self.scale_xy = 1.0/dboxes.scale_xy
        self.scale_wh = 1.0/dboxes.scale_wh

        self.sl1_loss = nn.SmoothL1Loss(reduction='none')
        self.dboxes = nn.Parameter(dboxes(order="xywh").transpose(0, 1).unsqueeze(dim = 0),
            requires_grad=False)
        # Two factor are from following links
        # http://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html
        self.con_loss = nn.CrossEntropyLoss(reduction='none')

    def _loc_vec(self, loc):
        """
            Generate Location Vectors
        """
        gxy = self.scale_xy*(loc[:, :2, :] - self.dboxes[:, :2, :])/self.dboxes[:, 2:, ]
        gwh = self.scale_wh*(loc[:, 2:, :]/self.dboxes[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self, ploc, plabel, gloc, glabel):
        """
            ploc, plabel: Nx4x8732, Nxlabel_numx8732
                predicted location and labels

            gloc, glabel: Nx4x8732, Nx8732
                ground truth location and labels
        """
        mask = glabel > 0
        pos_num = mask.sum(dim=1)

        vec_gd = self._loc_vec(gloc)

        # sum on four coordinates, and mask
        sl1 = self.sl1_loss(ploc, vec_gd).sum(dim=1)
        sl1 = (mask.float()*sl1).sum(dim=1)

        # hard negative mining
        con = self.con_loss(plabel, glabel)

        # postive mask will never selected
        con_neg = con.clone()
        con_neg[mask] = 0
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)

        # number of negative three times positive
        neg_num = torch.clamp(3*pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = con_rank < neg_num

        #print(con.shape, mask.shape, neg_mask.shape)
        closs = (con*((mask + neg_mask).float())).sum(dim=1)

        # avoid no object detected
        total_loss = sl1 + closs
        num_mask = (pos_num > 0).float()
        pos_num = pos_num.float().clamp(min=1e-6)
        ret = (total_loss*num_mask/pos_num).mean(dim=0)
        return ret


class SNAS(nn.Module):
    def __init__(self, backbone=ResNet_NAS('resnet50_nas')):
        super().__init__()

        self.feature_extractor = backbone

        self.label_num = 81  # number of COCO classes
        self._build_additional_features(self.feature_extractor.out_channels)
        self.num_defaults = [4, 6, 6, 6, 4, 4]
        self.loc = []
        self.conf = []

        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            self.loc.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            self.conf.append(nn.Conv2d(oc, nd * self.label_num, kernel_size=3, padding=1))

        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)
        self._init_weights()


    def _build_additional_features(self, input_size):
        self.additional_blocks = []
        for i, (input_size, output_size, channels) in enumerate(zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128])):
            if i < 3:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )
            else:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )

            self.additional_blocks.append(layer)

        self.additional_blocks = nn.ModuleList(self.additional_blocks)

    def _init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1: nn.init.xavier_uniform_(param)

    # Shape the classifier to the view of bboxes
    def bbox_view(self, src, loc, conf):
        ret = []
        for s, l, c in zip(src, loc, conf):
            ret.append((l(s).reshape(s.size(0), 4, -1), c(s).reshape(s.size(0), self.label_num, -1)))

        locs, confs = list(zip(*ret))
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs

    def forward(self, x):
        x = self.feature_extractor(x)

        detection_feed = [x]
        for l in self.additional_blocks:
            x = l(x)
            detection_feed.append(x)

        # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)

        # For SSD 300, shall return nbatch x 8732 x {nlabels, nlocs} results
        return locs, confs


if __name__ == '__main__':
    honey = [1, 1, 1, 2, 8, 3, 10, 17, 7, 18, 14, 10, 10, 13, 12, 13, 13, 24, 7, 22, 22, 21, 36, 9, 9, 7, 19, 9, 8, 71, 25, 7, 69, 12, 18, 9, 320]
    # ssd300 = SNAS(backbone=ResNet_NAS('resnet50_nas', honey=honey))
    backbone = ResNet50_new(honey=honey)
    # print(self.feature_extractor)
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(nn.Sequential(*list(backbone.children())[:6]), inputs=(input,))
    print('1',flops, params)
    # print(ResNet_NAS('resnet50_nas').flops())
    # for idx, layer in enumerate(ssd300.children()):
    #     print(f"Layer {idx}: {layer}")
