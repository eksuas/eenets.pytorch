"""
Custom EENet model
"""
from torch import nn

__all__ = ['CustomEENet', 'eenet8']

class CustomEENet(nn.Module):
    """Custom EENet-8 model.

    This model has identity connections such as ResNets.
    However, EENet-8 is a very small CNN having 2-8 filters in its layers.
    """
    def __init__(self, starting_filter):
        super(CustomEENet, self).__init__()
        self.filter = starting_filter
        self.initblock = nn.Sequential(
            nn.Conv2d(1, self.filter, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.filter),
            nn.ReLU(inplace=True),
        )
        self.basicblock1 = nn.Sequential(
            nn.Conv2d(self.filter, self.filter, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.filter),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filter, self.filter, kernel_size=3, stride=1, padding=1),
        )
        self.basicblock2 = nn.Sequential(
            nn.BatchNorm2d(self.filter),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filter, self.filter*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.filter*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filter*2, self.filter*2, kernel_size=3, stride=1, padding=1),
        )
        self.basicblock3 = nn.Sequential(
            nn.BatchNorm2d(self.filter*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filter*2, self.filter*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.filter*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filter*4, self.filter*4, kernel_size=3, stride=1, padding=1),
        )
        self.finalblock = nn.Sequential(
            nn.BatchNorm2d(self.filter*4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.conv2d_6 = nn.Conv2d(self.filter, self.filter*2, kernel_size=1, stride=2, padding=0)
        self.conv2d_9 = nn.Conv2d(self.filter*2, self.filter*4, kernel_size=1, stride=2, padding=0)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.exit0_classifier = nn.Sequential(
            nn.Linear(self.filter, 10),
            nn.Softmax(dim=1),
        )
        self.exit1_classifier = nn.Sequential(
            nn.Linear(self.filter*2, 10),
            nn.Softmax(dim=1),
        )
        self.exit0_confidence = nn.Sequential(
            nn.Linear(self.filter, 1),
            nn.Sigmoid(),
        )
        self.exit1_confidence = nn.Sequential(
            nn.Linear(self.filter*2, 1),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.filter*4, 10),
            nn.Softmax(dim=1),
        )
        self.cost = [0.60, 0.97]

    def forward(self, x):
        x = self.initblock(x)
        residual = self.basicblock1(x)
        x = residual + x

        e_x = self.pool(x).view(-1, self.filter)
        pred_0 = self.exit0_classifier(e_x)
        conf_0 = self.exit0_confidence(e_x)
        if (not self.training and conf_0.item() > 0.5):
            return pred_0, 0, self.cost[0]

        residual = self.basicblock2(x)
        x = self.conv2d_6(x)
        x = residual + x

        e_x = self.pool(x).view(-1, self.filter*2)
        pred_1 = self.exit1_classifier(e_x)
        conf_1 = self.exit1_confidence(e_x)
        if (not self.training and conf_1.item() > 0.5):
            return pred_1, 1, self.cost[1]

        residual = self.basicblock3(x)
        x = self.conv2d_9(x)
        x = residual + x
        x = self.finalblock(x)
        x = x.view(-1, self.filter*4)
        pred_2 = self.classifier(x)
        if not self.training:
            return pred_2, 2, 1.0

        return (pred_0, pred_1, pred_2), (conf_0, conf_1), self.cost


def eenet8(filters=2):
    """EENet-8 model.

    This creates an instance of Custom EENet-8 with given starting filter number.
    """
    return CustomEENet(filters)
