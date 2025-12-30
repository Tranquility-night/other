import torch
import torch.nn as nn
import numpy as np

# 残差块：Darknet-53核心组件
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

# Darknet-53骨干网络：特征提取
class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.res_block1 = self._make_res_block(64, 64, 1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.res_block2 = self._make_res_block(128, 128, 2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.res_block3 = self._make_res_block(256, 256, 8)  # 输出52×52
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.res_block4 = self._make_res_block(512, 512, 8)  # 输出26×26
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.res_block5 = self._make_res_block(1024, 1024, 4)  # 输出13×13

    def _make_res_block(self, in_channels, out_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res_block1(out)
        out = self.conv3(out)
        out = self.res_block2(out)
        out = self.conv4(out)
        out52 = self.res_block3(out)  # 52×52特征图
        out = self.conv5(out52)
        out26 = self.res_block4(out)  # 26×26特征图
        out = self.conv6(out26)
        out13 = self.res_block5(out)  # 13×13特征图
        return out52, out26, out13

# YOLO-V3检测头：特征融合+预测
class YOLOV3(nn.Module):
    def __init__(self, num_classes=20):
        super(YOLOV3, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = 3  # 每个尺度3个锚框
        self.darknet53 = Darknet53()

        # 13×13尺度检测分支（大目标）
        self.conv13_1 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv13_2 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv13_3 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv13_4 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv13_5 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv13_pred = nn.Conv2d(512, self.num_anchors * (5 + num_classes), 1, 1, 0)

        # 上采样+26×26尺度检测分支（中目标）
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.conv26_1 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, 1, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv26_2 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv26_3 = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv26_4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv26_5 = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv26_pred = nn.Conv2d(256, self.num_anchors * (5 + num_classes), 1, 1, 0)

        # 上采样+52×52尺度检测分支（小目标）
        self.conv_up2 = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.conv52_1 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 1, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv52_2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv52_3 = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv52_4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv52_5 = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv52_pred = nn.Conv2d(128, self.num_anchors * (5 + num_classes), 1, 1, 0)

    def forward(self, x):
        # 骨干网络提取多尺度特征
        feat52, feat26, feat13 = self.darknet53(x)

        # 13×13尺度预测
        out13 = self.conv13_1(feat13)
        out13 = self.conv13_2(out13)
        out13 = self.conv13_3(out13)
        out13 = self.conv13_4(out13)
        out13_feat = self.conv13_5(out13)
        pred13 = self.conv13_pred(out13_feat)  # [B, 3*(5+C), 13, 13]

        # 26×26尺度预测
        up1 = self.conv_up1(out13_feat)
        out26 = torch.cat([up1, feat26], dim=1)
        out26 = self.conv26_1(out26)
        out26 = self.conv26_2(out26)
        out26 = self.conv26_3(out26)
        out26 = self.conv26_4(out26)
        out26_feat = self.conv26_5(out26)
        pred26 = self.conv26_pred(out26_feat)  # [B, 3*(5+C), 26, 26]

        # 52×52尺度预测
        up2 = self.conv_up2(out26_feat)
        out52 = torch.cat([up2, feat52], dim=1)
        out52 = self.conv52_1(out52)
        out52 = self.conv52_2(out52)
        out52 = self.conv52_3(out52)
        out52 = self.conv52_4(out52)
        out52_feat = self.conv52_5(out52)
        pred52 = self.conv52_pred(out52_feat)  # [B, 3*(5+C), 52, 52]

        # 调整输出形状：[B, H, W, 3, 5+C]
        batch_size = x.size(0)
        h13, w13 = pred13.size(2), pred13.size(3)
        h26, w26 = pred26.size(2), pred26.size(3)
        h52, w52 = pred52.size(2), pred52.size(3)

        pred13 = pred13.permute(0, 2, 3, 1).reshape(batch_size, h13, w13, self.num_anchors, 5 + self.num_classes)
        pred26 = pred26.permute(0, 2, 3, 1).reshape(batch_size, h26, w26, self.num_anchors, 5 + self.num_classes)
        pred52 = pred52.permute(0, 2, 3, 1).reshape(batch_size, h52, w52, self.num_anchors, 5 + self.num_classes)

        return pred52, pred26, pred13