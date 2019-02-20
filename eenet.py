import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from flops_counter import get_model_complexity_info
from resnet import *

__all__ = ['EENet',
           'eenet18', 'eenet34', 'eenet50', 'eenet101', 'eenet152',
           'eenet20', 'eenet32', 'eenet44', 'eenet56',  'eenet110',]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ExitBlock(nn.Module):
    def __init__(self, inplanes, num_classes):
        super(ExitBlock, self).__init__()
        self.inplanes = inplanes
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.confidence = nn.Sequential(
            nn.Linear(inplanes, 1),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(inplanes, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.pool(x).view(-1, self.inplanes)
        h = self.confidence(x)
        y = self.classifier(x)
        return y, h


class EENet(nn.Module):

    def __init__(self, is_6n2model, block, total_layers, layer_conf=None, num_ee=3,
                 distribution="pareto", num_classes=1000, zero_init_residual=False,
                 input_shape=(3, 32, 32), **kwargs):
        super(EENet, self).__init__()
        self.inplanes = 64
        if is_6n2model:
            self.inplanes = 16
            layer_conf = [(total_layers-2) // 6]*3

        total_flops, total_params = get_model_complexity_info(
                eval("resnet"+str(total_layers)+"(num_classes="+str(num_classes)+")"),
                input_shape, print_per_layer_stat=False, as_strings=False)

        gold_rate = 1.61803398875
        flop_margin = 1.0 / (num_ee+1)
        threshold = []
        for e in range(num_ee):
            if distribution == "pareto":
                threshold.append(total_flops * (1 - (0.8**(e+1))))
            elif distribution == "fine":
                threshold.append(total_flops * (1 - (0.95**(e+1))))
            elif distribution == "linear":
                threshold.append(total_flops * flop_margin * (e+1))
            else:
                threshold.append(total_flops * (gold_rate**(e - num_ee)))

        self.stages = nn.ModuleList()
        self.exits = nn.ModuleList()
        self.cost = []
        self.complexity = []
        layers = nn.ModuleList()
        stage_id = 0

        if is_6n2model:
            layers.append(nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
            ))
        else:
            layers.append(nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ))

        planes = self.inplanes
        stride = 1
        for i in range(len(layer_conf)):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion),
                )

            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion

            part = nn.Sequential(*(list(self.stages)+list(layers)))
            flops, params = get_model_complexity_info(part,
                    input_shape, print_per_layer_stat=False, as_strings=False)
            if (stage_id < num_ee and flops >= threshold[stage_id]):
                self.stages.append(nn.Sequential(*layers))
                self.exits.append(ExitBlock(self.inplanes, num_classes))
                self.cost.append(flops / total_flops)
                self.complexity.append((flops, params))
                layers = nn.ModuleList()
                stage_id += 1


            for _ in range(1, layer_conf[i]):
                layers.append(block(self.inplanes, planes))

                part = nn.Sequential(*(list(self.stages)+list(layers)))
                flops, params = get_model_complexity_info(part,
                        input_shape, print_per_layer_stat=False, as_strings=False)
                if (stage_id < num_ee and flops >= threshold[stage_id]):
                    self.stages.append(nn.Sequential(*layers))
                    self.exits.append(ExitBlock(planes, num_classes))
                    self.cost.append(flops / total_flops)
                    self.complexity.append((flops, params))
                    layers = nn.ModuleList()
                    stage_id += 1

            planes *= 2
            stride = 2

        self.complexity.append((total_flops, total_params))

        if is_6n2model:
            layers.append(nn.AvgPool2d(8))
            self.fc = nn.Linear(64 * block.expansion, num_classes)
        else:
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.stages.append(nn.Sequential(*layers))
        self.softmax = nn.Softmax(dim=1)

        assert len(self.exits) == num_ee, \
            "The desired number of exit blocks is too much for the model capacity."

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        pred, conf = [], []

        for id, exitblock in enumerate(self.exits):
            x = self.stages[id](x)
            y, h = exitblock(x)
            if (not self.training and h.item() > 0.5):
                return y, id, self.cost[id]
            pred.append(y)
            conf.append(h)

        x = self.stages[-1](x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        y = self.softmax(x)
        if (not self.training):
            return y, len(self.exits), 1.0
        pred.append(y)

        return (pred, conf, self.cost)


def eenet18(**kwargs):
    model = EENet(False, BasicBlock, 18, [2, 2, 2, 2], **kwargs)
    return model

def eenet34(**kwargs):
    model = EENet(False, BasicBlock, 34, [3, 4, 6, 3], **kwargs)
    return model

def eenet50(**kwargs):
    model = EENet(False, Bottleneck, 50, [3, 4, 6, 3], **kwargs)
    return model

def eenet101(**kwargs):
    model = EENet(False, Bottleneck, 101, [3, 4, 23, 3], **kwargs)
    return model

def eenet152(**kwargs):
    model = EENet(False, Bottleneck, 152, [3, 8, 36, 3], **kwargs)
    return model

def eenet20(**kwargs):
    model = EENet(True, BasicBlock, 20, **kwargs)
    return model

def eenet32(**kwargs):
    model = EENet(True, BasicBlock, 32, **kwargs)
    return model

def eenet44(**kwargs):
    model = EENet(True, BasicBlock, 44, **kwargs)
    return model

def eenet56(**kwargs):
    model = EENet(True, BasicBlock, 56, **kwargs)
    return model

def eenet110(**kwargs):
    model = EENet(True, BasicBlock, 110, **kwargs)
    return model
