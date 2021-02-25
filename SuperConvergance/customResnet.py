import torch.nn as nn
import torch.nn.functional as F
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, padding=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        return out


class XBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, padding=1):
        super(XBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.mp1 = nn.MaxPool2d(kernel_size=2)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.bn1(self.mp1((self.conv1(x))))
        out = F.relu(out)
        return out


class XResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(XResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.mp1 = nn.MaxPool2d(kernel_size=4)
        self.layer1 = self._make_layer(
            [XBlock, BasicBlock], 128, stride=1, padding=1)
        self.layer2 = self._make_layer([XBlock], 256, stride=1, padding=1)
        self.layer3 = self._make_layer(
            [XBlock, BasicBlock], 512, stride=1, padding=1)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, blocks, planes, stride, padding):
        layers = []
        for block in blocks:
            layers.append(block(self.in_planes, planes, stride, padding))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # prep layer
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.mp1(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.softmax(out, dim=1)


def myResNet():
    return XResNet()
