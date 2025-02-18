import torch
import torch.nn as nn

class CCFM(nn.Module):
    def __init__(self, in_channels):
        super(CCFM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = self.relu(self.bn(x1 + x2))
        return out

class SENetv2(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SENetv2, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se_weight = self.global_pool(x)
        se_weight = self.relu(self.fc1(se_weight))
        se_weight = self.sigmoid(self.fc2(se_weight))
        return x * se_weight

class FusionCCFM_SE(nn.Module):
    def __init__(self, in_channels):
        super(FusionCCFM_SE, self).__init__()
        self.ccfm = CCFM(in_channels)
        self.senet = SENetv2(in_channels)

    def forward(self, x):
        x = self.ccfm(x)
        x = self.senet(x)
        return x
