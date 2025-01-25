import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log

conv_num_cfg = {
    'resnet18': 8,
    'resnet34': 16,
    'resnet50': 16,
    'resnet101': 33,
    'resnet152': 50
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, honey, index, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, int(planes * honey[index] / 10), kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(int(planes * honey[index] / 10))
        self.conv2 = nn.Conv2d(int(planes * honey[index] / 10), planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, honey, index, stride=1):
        super(Bottleneck, self).__init__()
        pr_channels = int(planes * honey[index] / 10)
        self.conv1 = nn.Conv2d(in_planes, pr_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(pr_channels)
        self.conv2 = nn.Conv2d(pr_channels, pr_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(pr_channels)
        self.conv3 = nn.Conv2d(pr_channels, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, honey=None):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.honey = honey
        self.current_conv = 0

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layers(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layers(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layers(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layers(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.Sequential(nn.AvgPool2d(7))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes,
                                self.honey, self.current_conv, stride))
            self.current_conv += 1
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class Bottleneck_bn(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, flag=None):
        super(Bottleneck_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes[0], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.conv3 = nn.Conv2d(planes[1], planes[2], kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes[2])

        self.downsample = nn.Sequential()
        if flag:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes[2], kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes[2])
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += identity
        out = F.relu(out)
        return out


def decode(honey, food_scale):
    t1 = [3,4,6,3]
    s1 = honey[1:33]
    s2 = honey[33:37]
    o1 = [0] * 48
    index1 = 0
    index2 = 0
    index3 = 0
    for i in range(48):
        if (i+1)%3 != 0:
            o1[i] = s1[index1]
            index1 += 1
        else:
            index3 += 1
            # print("3-------:%s"%index3)
            o1[i] = s2[index2]
            if index3 == t1[index2]:
                # print("2---------:%s"%index2)
                index3 = 0
                index2 += 1

    honey_new = [honey[0]] + o1
    # print(honey_new, len(honey_new))
    struc = []
    for i in range(len(food_scale)):
        struc.append(int(food_scale[i] * honey_new[i] / (21 * 2 ** int(log(food_scale[i], 2) - log(food_scale[0], 2)))))
    # print(struc, len(struc))
    return struc


class ResNet_new(nn.Module):
    food_scale = [64, 64, 64, 256, 64, 64, 256, 64, 64, 256, 128, 128, 512, 128, 128, 512, 128, 128, 512, 128, 128,
                  512, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256,
                  1024, 512, 512, 2048, 512, 512, 2048, 512, 512, 2048]
    def __init__(self, block, nb, num_classes=10, honey=None, food_scale=None):
        super(ResNet_new, self).__init__()
        # if honey is None:
        #     honey = [10, 10, 10, 40, 10, 10, 40, 10, 10, 40, 20, 20, 80, 20, 20, 80, 20, 20, 80, 20, 20, 80, 40, 40,
        #              160,40, 40, 160, 40, 40, 160, 40, 40, 160, 40, 40, 160, 40, 40, 160, 80, 80, 320, 80, 80, 320, 80, 80,
        #              320]
        if honey is None:
            self.struc = self.food_scale
        else:
            self.struc = decode(honey, self.food_scale)

        self.in_planes = self.struc[0]
        self.conv1 = nn.Conv2d(3, self.struc[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.struc[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layers(block, self.struc[1 : 3*nb[0]+1], nb[0],  stride=1)
        self.layer2 = self._make_layers(block, self.struc[3*nb[0]+1 : 3*(nb[0]+nb[1])+1], nb[1],  stride=2)
        self.layer3 = self._make_layers(block, self.struc[3*(nb[0]+nb[1])+1: 3*(nb[0]+nb[1]+nb[2])+1], nb[2],  stride=2)
        self.layer4 = self._make_layers(block, self.struc[3*(nb[0]+nb[1]+nb[2])+1 : 3*(nb[0]+nb[1]+nb[2]+nb[3])+1], nb[3],  stride=2)

        self.avgpool = nn.Sequential(nn.AvgPool2d(7))
        self.fc = nn.Linear(self.struc[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1) #3 [1,0,0] 4:[1, 0, 0, 0] 6:[1, 0, 0, 0, 0, 0] 3:[1, 0, 0, 0]
        flag = [1] + [0] * (num_blocks - 1)
        layers = []
        for n in range(len(strides)):
            layers.append(block(self.in_planes, planes[3*n:3*(n+1)], strides[n], flag[n]))
            self.in_planes = planes[3*(n+1)-1]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def resnet(cfg, honey=None, num_classes=1000):
    if honey == None:
        honey = conv_num_cfg[cfg] * [10]
    if cfg == 'resnet18':
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, honey=honey)
    elif cfg == 'resnet34':
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, honey=honey)
    elif cfg == 'resnet50':
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, honey=honey)
    elif cfg == 'resnet101':
        return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, honey=honey)
    elif cfg == 'resnet152':
        return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, honey=honey)


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    #[3, 12, 24, 42]
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1000, honey=None)

def ResNet50_new(honey=None,food_scale=None):
    #[3, 12, 24, 42]
   return ResNet_new(Bottleneck_bn, [3, 4, 6, 3], num_classes=1000, honey=honey, food_scale=food_scale)


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])

def test1():
    honey1 = []
    honey2 = []
    # print(len(honey))
    model = resnet('resnet50')
    print(model)
    # for name, layer in model.named_modules():
    #     if 'downsample' in name and isinstance(layer, nn.BatchNorm2d):
    #         continue
    #     if isinstance(layer, nn.BatchNorm2d):
    #         size = layer.weight.data.shape[0]
    #         print(size)
    #         honey1.append(size)
    #
    # print(len(honey1))
    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         size = m.weight.data.shape[0]
    #         honey2.append(size)
    # print(honey1)
    # print(honey2)
    # print(len(honey1), len(honey2))
    # print(sum(honey2))




def test():
    honey = [10, 8, 10, 8, 6, 5, 2, 10, 16, 17, 4, 13, 15, 20, 12, 14, 23, 10, 15, 19, 33, 33, 33, 40, 28, 35, 28, 67, 31, 61, 80, 8, 45, 15, 47, 160, 233]
    model = ResNet50_new(honey)
    origin_model = resnet('resnet50')
    from thop import profile
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input,))
    oriflops, oriparams = profile(origin_model, inputs=(input,))
    print(flops, params)
    print('FLOPS Compress Rate: %.2f M/%.2f M(%.2f%%)' % (
    flops / 1000000, oriflops / 1000000, 100. * (oriflops - flops) / oriflops))
    pass
                 # [38, 19, 19, 44, 89, 51, 44, 51, 204, 44, 38, 25, 25, 19, 96, 25, 83, 294, 25, 44, 19, 25, 134, 25, 192, 25, 192, 192, 25, 19, 192, 294, 6, 192, 147, 672, 192, 185, 108, 192, 307, 153, 633, 96, 454, 633, 198, 96, 633]
def test2():
    food_scale = [64, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256,
                       256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 256, 512, 1024, 2048]
    print(len(food_scale))
    honey = [7, 10, 10, 6, 10, 9, 6, 16, 11, 14, 5, 10, 7, 12, 13, 27, 9, 14, 11, 21, 40, 32, 14, 27, 21, 17, 29, 53, 65, 14, 28, 52, 73, 39, 70, 147, 220]

    struc = []
    for i in range(len(food_scale)):
        struc.append(int(food_scale[i] * honey[i] / (10 * 2 ** int(log(food_scale[i], 2) - log(food_scale[0], 2)))))
    # decode(honey, food_scale)
    print(struc)

#[64, 64, 64, 256, 64, 64, 256, 64, 64, 256, 128, 128, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 512, 512, 2048, 512, 512, 2048, 512, 512, 2048, 256, 512, 1024, 2048]

if __name__ == '__main__':
    test2()


