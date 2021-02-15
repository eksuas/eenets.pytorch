"""
Edanur Demir
Custom EENet model
"""
from torch import nn

__all__ = ['CustomEENet', 'eenet8']

class CustomEENet(nn.Module):
    """Custom EENet-8 model.

    This model (EENet-8) consists of constant two early-exit blocks.
    and it is a very small CNN having 2-8 filters in its layers.
    """
    def __init__(self, input_shape, num_classes, starting_filter):
        super(CustomEENet, self).__init__()
        channel, _, _ = input_shape
        self.filter = starting_filter
        self.initblock = nn.Sequential(
            nn.Conv2d(channel, self.filter, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.filter),
            nn.ReLU(inplace=True),
        )
        self.basicblock1 = nn.Sequential(
            nn.Conv2d(self.filter, self.filter, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.filter),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filter, self.filter, kernel_size=3, stride=1, padding=1),
        )
        self.basicblock2 = self.get_basic_block(1)
        self.basicblock3 = self.get_basic_block(2)
        self.finalblock = nn.Sequential(
            nn.BatchNorm2d(self.filter*4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.conv2d_6 = nn.Conv2d(self.filter, self.filter*2, kernel_size=1, stride=2, padding=0)
        self.conv2d_9 = nn.Conv2d(self.filter*2, self.filter*4, kernel_size=1, stride=2, padding=0)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.exit0_classifier = self.get_classifier(num_classes, 1)
        self.exit1_classifier = self.get_classifier(num_classes, 2)
        self.exit0_confidence = self.get_confidence(1)
        self.exit1_confidence = self.get_confidence(2)
        self.classifier = self.get_classifier(num_classes, 4)
        self.complexity = [(546, 137), (1844, 407), (6982, 1490)]
        if self.filter == 4:
            self.complexity = [(1792, 407), (6608, 1395), (25814, 5332)]

    def get_basic_block(self, expansion):
        """get basic block as nn.Sequential"""
        filter_in = self.filter * expansion
        return nn.Sequential(
            nn.BatchNorm2d(filter_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_in, filter_in*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filter_in*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_in*2, filter_in*2, kernel_size=3, stride=1, padding=1))

    def get_classifier(self, num_classes, expansion):
        """get classifier as nn.Sequential"""
        filter_in = self.filter * expansion
        return nn.Sequential(
            nn.Linear(filter_in, num_classes),
            nn.Softmax(dim=1))

    def get_confidence(self, expansion):
        """get confidence as nn.Sequential"""
        filter_in = self.filter * expansion
        return nn.Sequential(
            nn.Linear(filter_in, 1),
            nn.Sigmoid())

    def forward(self, x):
        cost_0, cost_1 = 0.08, 0.26
        x = self.initblock(x)
        residual = self.basicblock1(x)
        x = residual + x

        e_x = self.pool(x).view(-1, self.filter)
        pred_0 = self.exit0_classifier(e_x)
        conf_0 = self.exit0_confidence(e_x)
        if (not self.training and conf_0.item() > 0.5):
            return pred_0, 0, cost_0

        residual = self.basicblock2(x)
        x = self.conv2d_6(x)
        x = residual + x

        e_x = self.pool(x).view(-1, self.filter*2)
        pred_1 = self.exit1_classifier(e_x)
        conf_1 = self.exit1_confidence(e_x)
        if (not self.training and conf_1.item() > 0.5):
            return pred_1, 1, cost_1

        residual = self.basicblock3(x)
        x = self.conv2d_9(x)
        x = residual + x

        x = self.finalblock(x)
        x = x.view(-1, self.filter*4)
        pred_2 = self.classifier(x)
        if not self.training:
            return pred_2, 2, 1.0

        return (pred_0, pred_1, pred_2), (conf_0, conf_1), (cost_0, cost_1)


def eenet8(input_shape, num_classes, filters=2, **kwargs):
    """EENet-8 model.

    This creates an instance of Custom EENet-8 with given starting filter number.
    """
    print('Note that EENet-8 has constant two early-exit blocks regardless of what num_ee is!')
    return CustomEENet(input_shape, num_classes, filters)
