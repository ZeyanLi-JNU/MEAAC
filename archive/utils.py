import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderWithAttention(nn.Module):
    def __init__(self, in_channels, feature_dim):
        super(EncoderWithAttention, self).__init__()
        self.msf1 = MSFEBWithAttention(in_channels, 63)
        self.dsc1 = DepthwiseSeparableConv(63, 128, 3, 2, 1)
        self.msf2 = MSFEBWithAttention(128, 252)
        self.dsc2 = DepthwiseSeparableConv(252, 512, 3, 2, 1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, feature_dim)

    def forward(self, x):
        x = self.msf1(x)
        x = self.dsc1(x)
        x = self.msf2(x)
        x = self.dsc2(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class MSFEBWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSFEBWithAttention, self).__init__()
        # Ensure out_channels is divisible by 3
        assert out_channels % 3 == 0, "out_channels must be divisible by 3"

        self.conv1x1 = nn.Conv2d(in_channels, out_channels // 3, kernel_size=1, stride=1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels // 3, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels // 3, kernel_size=5, stride=1, padding=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_channels)

    def forward(self, x):
        x1 = self.conv1x1(x)
        x3 = self.conv3x3(x)
        x5 = self.conv5x5(x)
        out = torch.cat([x1, x3, x5], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        out = self.ca(out)
        return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EnhancedContrastiveLoss(nn.Module):
    def __init__(self, margin, alpha=1.0, beta=1.0):
        super(EnhancedContrastiveLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta

    def forward(self, anchor, positive, negative):
        positive_distance = F.pairwise_distance(anchor, positive, p=2)
        negative_distance = F.pairwise_distance(anchor, negative, p=2)
        contrastive_loss = F.relu(positive_distance - negative_distance + self.margin)
        enhanced_term = torch.pow(F.relu(negative_distance - self.margin), 2)
        loss = self.alpha * contrastive_loss.mean() + self.beta * enhanced_term.mean()
        return loss