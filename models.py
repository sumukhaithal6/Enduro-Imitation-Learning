#!/usr/bin/env python3
"""
Model architecture.

Authors:
LICENCE:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    """Test net."""

    def __init__(self):
        """Ctor."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(29008, 120)
        self.fc1 = nn.Linear(21904, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 9)

    def forward(self, x):
        """Forward call."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BigNet(nn.Module):
    """Test net."""

    def __init__(self):
        """Ctor."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)

        # self.fc1 = nn.Linear(29008, 120)
        self.fc1 = nn.Linear(8192, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 9)

    def forward(self, x):
        """Forward call."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ResNet(nn.Module):
    """
    ResNet in PyTorch.

    For Pre-activation ResNet, see 'preact_resnet.py'.

    Reference:
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Deep Residual Learning for Image Recognition. arXiv:1512.03385
    """

    class BasicBlock(nn.Module):
        """Basic block."""

        expansion = 1

        def __init__(self, in_planes, planes, stride=1):
            """Ctor."""
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(
                planes, planes, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn2 = nn.BatchNorm2d(planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

        def forward(self, x):
            """Forward call."""
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

    class Bottleneck(nn.Module):
        """Bottlenect."""

        expansion = 4

        def __init__(self, in_planes, planes, stride=1):
            """Ctor."""
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_planes,
                planes,
                kernel_size=1,
                bias=False,
            )
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(
                planes, self.expansion * planes, kernel_size=1, bias=False
            )
            self.bn3 = nn.BatchNorm2d(self.expansion * planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

        def forward(self, x):
            """Forward call."""
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

    def __init__(self, block, num_blocks, num_classes=9):
        """Ctor."""
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(12800, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward call."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.linear(out)
        return out


def ResNet18() -> ResNet:
    """Return ResNet 18."""
    return ResNet(ResNet.BasicBlock, [2, 2, 2, 2])


def ResNet34() -> ResNet:
    """Return ResNet 34."""
    return ResNet(ResNet.BasicBlock, [3, 4, 6, 3])


def ResNet50() -> ResNet:
    """Return ResNet 50."""
    return ResNet(ResNet.Bottleneck, [3, 4, 6, 3])
