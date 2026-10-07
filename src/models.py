"""ResNet-18 cho CIFAR-10, có lớp Normalize đặt ở đầu mạng.

Đầu vào của mô hình luôn là ảnh trong khoảng [0, 1]; việc chuẩn hoá mean/std
được thực hiện bên trong mô hình để epsilon của tấn công có nghĩa trên thang pixel.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CIFAR10_MEAN, CIFAR10_STD, NUM_CLASSES


class Normalize(nn.Module):
    """Chuẩn hoá (x - mean) / std theo từng kênh. mean/std là buffer nên được lưu trong state_dict."""

    def __init__(self, mean=CIFAR10_MEAN, std=CIFAR10_STD):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    """ResNet bản CIFAR: conv đầu 3x3 stride 1, không có maxpool."""

    def __init__(self, block, num_blocks, num_classes=NUM_CLASSES):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, n, stride):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.flatten(self.avgpool(out), 1)
        return self.fc(out)


def ResNet18(num_classes=NUM_CLASSES):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def build_model(num_classes=NUM_CLASSES) -> nn.Sequential:
    """Mô hình dùng chung cho cả project: nhận ảnh [0,1], trả về logits."""
    return nn.Sequential(Normalize(), ResNet18(num_classes))
