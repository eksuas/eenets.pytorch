import torch
import torch.nn as nn
import torchvision.models as models

class EENet(nn.Module):
	def __init__(self):
		super(EENet, self).__init__()
        self.initblock = nn.Sequential(
			nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(4),
			nn.ReLU(inplace=True),
		)
		self.basicblock1 = nn.Sequential(
			nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(4),
			nn.ReLU(inplace=True),
			nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
		)
		self.basicblock2 = nn.Sequential(
			nn.BatchNorm2d(4),
			nn.ReLU(inplace=True),
			nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(8),
			nn.ReLU(inplace=True),
			nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
		)
		self.basicblock3 = nn.Sequential(
			nn.BatchNorm2d(8),
			nn.ReLU(inplace=True),
			nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(16),
			nn.ReLU(inplace=True),
			nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
		)
		self.finalblock = nn.Sequential(
			nn.BatchNorm2d(16),
			nn.ReLU(inplace=True),
			nn.AdaptiveAvgPool2d(1),
		)
		self.conv2d_6 = nn.Conv2d(4, 8, kernel_size=1, stride=2, padding=0)
		self.conv2d_9 = nn.Conv2d(8, 16, kernel_size=1, stride=2, padding=0)
		self.pool = nn.AdaptiveAvgPool2d(1)

		self.exit0_classifier = nn.Sequential(
			nn.Linear(4, 10),
			nn.Softmax(dim=1),
		)
		self.exit1_classifier = nn.Sequential(
			nn.Linear(8, 10),
			nn.Softmax(dim=1),
		)
		self.exit0_confidence = nn.Sequential(
			nn.Linear(4, 1),
			nn.Sigmoid(),
		)
		self.exit1_confidence = nn.Sequential(
			nn.Linear(8, 1),
			nn.Sigmoid(),
		)
		self.classifier = nn.Sequential(
			nn.Linear(16, 10),
			nn.Softmax(dim=1),
		)

	def forward(self, x):
  		x = self.initblock(x)
        residual = self.basicblock1(x)
        x = residual + x

        e_x = self.pool(x).view(-1, 4)
        y0 = self.exit0_classifier(e_x)
        h0 = self.exit0_confidence(e_x)

        if (not self.training and torch.mean(h0) > 0.5):
            return y0, 0, 0.08

        residual = self.basicblock2(x)
        x = self.conv2d_6(x)
        x = residual + x

        e_x = self.pool(x).view(-1, 8)
        y1 = self.exit1_classifier(e_x)
        h1 = self.exit1_confidence(e_x)
        if (not self.training and torch.mean(h1) > 0.5):
            return y1, 1, 0.26

        residual = self.basicblock3(x)
        x = self.conv2d_9(x)
        x = residual + x
        x = self.finalblock(x)
        x = x.view(-1, 16)
        y2 = self.classifier(x)

        if (not self.training):
            return y2, 2, 1.00

        return (y0, y1, y2), (h0, h1)
