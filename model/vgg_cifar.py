import torch.nn as nn
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from math import log

cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

defaultcfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

#honeysource: 1d向量，值限制在0-9

class BeeVGG(nn.Module):
    def __init__(self, vgg_name, honeysource, food_scale):
        super(BeeVGG, self).__init__()
        self.honeysource = honeysource
        self.food_scale = food_scale
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(int(512 * honeysource[len(honeysource)-1] / (10 * 2 ** int(log(food_scale[-1], 2)-log(food_scale[0], 2)))), 10)
        # self.classifier = nn.Linear(honeysource[len(honeysource) - 1], 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
        
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        index = 0
        Mlayers = 0
        for x_index, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                Mlayers += 1
            else:

                x = int(x * self.honeysource[x_index - Mlayers] / (10 * 2 ** int(log(self.food_scale[x_index - Mlayers], 2)-log(self.food_scale[0], 2))))
                if x == 0:
                    x = 1
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class VGG_ori(nn.Module):
    def __init__(self, cfg, dataset='cifar10', depth='vgg16', init_weights=True):
        super(VGG_ori, self).__init__()

        if cfg is None:
            cfg = defaultcfg[depth]
        else:
            cfg = defaultcfg[cfg]


        self.feature = self.make_layers(cfg, True)

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        self.classifier = nn.Linear(cfg[-1], num_classes)
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        Mlayers = 0

        for x_index, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                Mlayers += 1
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        # x = x.view(-1, 1*1*17)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class VGG_new(nn.Module):
    def __init__(self, depth, honeysource=None, food_scale=[64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512], dataset='cifar10', init_weights=True):
        super(VGG_new, self).__init__()
        self.honeysource = honeysource
        self.food_scale = food_scale
        if honeysource is None:
            cfg = defaultcfg[depth]
            honeysource= [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
            self.honeysource = food_scale

        self.feature = self.make_layers(defaultcfg[depth], True)

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        self.classifier = nn.Linear(int(512 * honeysource[len(honeysource)-1] / (10 * 2 ** int(log(food_scale[-1], 2)-log(food_scale[0], 2)))), num_classes)
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        Mlayers = 0

        for x_index, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                Mlayers += 1
            else:
                v = int(v * self.honeysource[x_index - Mlayers] / (10 * 2 ** int(log(self.food_scale[x_index - Mlayers], 2) - log(self.food_scale[0], 2))))
                if v == 0:
                    v = 1
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        # x = x.view(-1, 1*1*17)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


#honeysource: 1d向量，值根据层通道数不同，限制在不同的范围
class BeeVGG_new(nn.Module):
    def __init__(self, vgg_name, honeysource):
        super(BeeVGG, self).__init__()
        self.honeysource = honeysource
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(int(512 * honeysource[len(honeysource) - 1]), 10)
        # self.classifier = nn.Linear(honeysource[len(honeysource) - 1], 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        index = 0
        Mlayers = 0
        for x_index, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                Mlayers += 1
            else:
                x = int(x * self.honeysource[x_index - Mlayers])
                if x == 0:
                    x = 1
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

if __name__ == '__main__':
    from thop import profile

    input = torch.randn(1, 3, 32, 32)
    origin_model = VGG_ori('vgg16')
    oriflops, params = profile(origin_model, inputs=(input,))
    model = VGG_new('vgg16', honeysource=[3, 4, 7, 9, 20, 25, 20, 71, 17, 24, 14, 40, 54])
    print(model)
    flops, params = profile(model, inputs=(input,))
    print(flops)
    print('FLOPS Compress Rate: %.2f M/%.2f M(%.2f%%)' % (
        flops / 1000000, oriflops / 1000000, 100. * (oriflops - flops) / oriflops))

