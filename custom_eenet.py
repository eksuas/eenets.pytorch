import torch
import torch.nn as nn

__all__ = ['CustomEENet', 'eenet8']

class CustomEENet(nn.Module):
	def __init__(self, filter):
		super(CustomEENet, self).__init__()
		self.filter = filter
		self.initblock = nn.Sequential(
			nn.Conv2d(1, filter, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(filter),
			nn.ReLU(inplace=True),
		)
		self.basicblock1 = nn.Sequential(
			nn.Conv2d(filter, filter, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(filter),
			nn.ReLU(inplace=True),
			nn.Conv2d(filter, filter, kernel_size=3, stride=1, padding=1),
		)
		self.basicblock2 = nn.Sequential(
			nn.BatchNorm2d(filter),
			nn.ReLU(inplace=True),
			nn.Conv2d(filter, filter*2, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(filter*2),
			nn.ReLU(inplace=True),
			nn.Conv2d(filter*2, filter*2, kernel_size=3, stride=1, padding=1),
		)
		self.basicblock3 = nn.Sequential(
			nn.BatchNorm2d(filter*2),
			nn.ReLU(inplace=True),
			nn.Conv2d(filter*2, filter*4, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(filter*4),
			nn.ReLU(inplace=True),
			nn.Conv2d(filter*4, filter*4, kernel_size=3, stride=1, padding=1),
		)
		self.finalblock = nn.Sequential(
			nn.BatchNorm2d(filter*4),
			nn.ReLU(inplace=True),
			nn.AdaptiveAvgPool2d(1),
		)
		self.conv2d_6 = nn.Conv2d(filter, filter*2, kernel_size=1, stride=2, padding=0)
		self.conv2d_9 = nn.Conv2d(filter*2, filter*4, kernel_size=1, stride=2, padding=0)
		self.pool = nn.AdaptiveAvgPool2d(1)

		self.exit0_classifier = nn.Sequential(
			nn.Linear(filter, 10),
			nn.Softmax(dim=1),
		)
		self.exit1_classifier = nn.Sequential(
			nn.Linear(filter*2, 10),
			nn.Softmax(dim=1),
		)
		self.exit0_confidence = nn.Sequential(
			nn.Linear(filter, 1),
			nn.Sigmoid(),
		)
		self.exit1_confidence = nn.Sequential(
			nn.Linear(filter*2, 1),
			nn.Sigmoid(),
		)
		self.classifier = nn.Sequential(
			nn.Linear(filter*4, 10),
			nn.Softmax(dim=1),
		)
		self.cost = [0.60, 0.97]

	def forward(self, x):
		x = self.initblock(x)
		residual = self.basicblock1(x)
		x = residual + x

		e_x = self.pool(x).view(-1, self.filter)
		y0 = self.exit0_classifier(e_x)
		h0 = self.exit0_confidence(e_x)
		if (not self.training and h0.item() > 0.5):
			return y0, 0, self.cost[0]

		residual = self.basicblock2(x)
		x = self.conv2d_6(x)
		x = residual + x

		e_x = self.pool(x).view(-1, self.filter*2)
		y1 = self.exit1_classifier(e_x)
		h1 = self.exit1_confidence(e_x)
		if (not self.training and h1.item() > 0.5):
			return y1, 1, self.cost[1]

		residual = self.basicblock3(x)
		x = self.conv2d_9(x)
		x = residual + x
		x = self.finalblock(x)
		x = x.view(-1, self.filter*4)
		y2 = self.classifier(x)
		if (not self.training):
			return y2, 2, 1.0

		return (y0, y1, y2), (h0, h1), self.cost


def eenet8(filters=2, **kwargs):
    model = CustomEENet(filters)
    return model
