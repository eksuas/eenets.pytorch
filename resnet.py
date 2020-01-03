"""
ResNet models
"""
from torch import  nn
from torch.utils import model_zoo
from flops_counter import get_model_complexity_info

__all__ = ['ResNet', 'ResNet6n2',
           'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
           'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',]


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """Basic Block defition.

    Basic 3X3 convolution blocks for use on ResNets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, **kwargs):
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
    """Bottleneck Block defition.

    Bottleneck architecture for > 34 layer ResNets.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, **kwargs):
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


class ResNet(nn.Module):
    """Builds a ResNet like architecture.

    Arguments are
    * block:              Block function of the architecture either 'BasicBlock' or 'Bottleneck'.
    * layers:             The total number of layers.
    * num_classes:        The number of classes in the dataset.
    * zero_init_residual: Zero-initialize the last BN in each residual branch,
                          so that the residual branch starts with zeros,
                          and each residual block behaves like an identity. This improves the model
                          by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

    Returns:
        The nn.Module.
    """
    def __init__(self, block, layers, num_classes, input_shape, zero_init_residual=False, **kwargs):
        super(ResNet, self).__init__()
        channel, _, _ = input_shape
        self.input_shape = input_shape
        self.inplanes = 64
        self.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fully_connected = nn.Linear(512 * block.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, Bottleneck):
                    nn.init.constant_(module.bn3.weight, 0)
                elif isinstance(module, BasicBlock):
                    nn.init.constant_(module.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def set_multiple_gpus(self):
        self = nn.DataParallel(self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)

        return x


class ResNet6n2(nn.Module):
    """Builds a ResNet like architecture.

    Arguments are
    * block:              Block function of the architecture either 'BasicBlock' or 'Bottleneck'.
    * layers:             The total number of layers.
    * num_classes:        The number of classes in the dataset.
    * zero_init_residual: Zero-initialize the last BN in each residual branch,
                          so that the residual branch starts with zeros,
                          and each residual block behaves like an identity. This improves the model
                          by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

    Returns:
        The nn.Module.
    """
    def __init__(self, block, layers, num_classes, input_shape, zero_init_residual=False, **kwargs):
        super(ResNet6n2, self).__init__()
        channel, _, _ = input_shape
        layer_blocks = (layers-2) // 6
        self.input_shape = input_shape
        self.inplanes = 16
        self.conv1 = nn.Conv2d(channel, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.stage1 = self._make_layer(block, 16, layer_blocks)
        self.stage2 = self._make_layer(block, 32, layer_blocks, stride=2)
        self.stage3 = self._make_layer(block, 64, layer_blocks, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fully_connected = nn.Linear(64 * block.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, Bottleneck):
                    nn.init.constant_(module.bn3.weight, 0)
                elif isinstance(module, BasicBlock):
                    nn.init.constant_(module.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def set_multiple_gpus(self):
        self = nn.DataParallel(self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)

        return x


def set_complexity(model):
    """get model complexity in terms of FLOPs and the number of parameters"""
    flops, params = get_model_complexity_info(model, model.input_shape,\
                    print_per_layer_stat=False, as_strings=False)
    model.complexity = [(flops, params)]


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Arguments are
    * pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    set_complexity(model)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Arguments are
    * pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    set_complexity(model)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Arguments are
    * pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    set_complexity(model)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Arguments are
    * pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    set_complexity(model)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Arguments are
    * pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    set_complexity(model)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def resnet20(**kwargs):
    """Constructs a ResNet-20 model."""
    model = ResNet6n2(BasicBlock, 20, **kwargs)
    set_complexity(model)
    return model


def resnet32(**kwargs):
    """Constructs a ResNet-32 model."""
    model = ResNet6n2(BasicBlock, 32, **kwargs)
    set_complexity(model)
    return model


def resnet44(**kwargs):
    """Constructs a ResNet-44 model."""
    model = ResNet6n2(BasicBlock, 44, **kwargs)
    set_complexity(model)
    return model


def resnet56(**kwargs):
    """Constructs a ResNet-56 model."""
    model = ResNet6n2(BasicBlock, 56, **kwargs)
    set_complexity(model)
    return model


def resnet110(**kwargs):
    """Constructs a ResNet-110 model."""
    model = ResNet6n2(BasicBlock, 110, **kwargs)
    set_complexity(model)
    return model
