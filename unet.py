import torch
import torch.nn as nn
import math

class UNet(nn.Module):
    def __init__(self, inplanes, planes, stride=1) -> None:
        super(UNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, 64, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128))
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128))
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256))
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(236))
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512))
        self.conv8 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512))
        self.conv9 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding='same'),
            nn.ReLU(inplace=True))
        self.conv10 = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024))
        self.conv11 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512))
        self.conv12 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512))
        self.conv13 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256))
        self.conv14 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256))
        self.conv15 =  nn.Sequential(
            nn.Conv2d(256, 128, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128))
        self.conv16 =  nn.Sequential(
            nn.Conv2d(128, 128, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128))
        self.conv17 =  nn.Sequential(
            nn.Conv2d(128, 64, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64))
        self.conv18 =  nn.Sequential(
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64))
        self.conv19 =  nn.Sequential(
            nn.Conv2d(64, planes, 1, padding='same'),
            nn.Sigmoid())

    def maxPool2dSame(self, x: torch.Tensor, kernelSize, stride = 2):
        _, _, inWidth, inHeight = x.size()
        outWidth = math.ceil(float(inWidth)/float(stride))
        outHeight = math.ceil(float(inHeight)/float(stride))
        padAlongWidth = max((outWidth - 1) * stride + kernelSize - inWidth,0)
        padAlongHeight = max((outHeight - 1) * stride + kernelSize - inHeight,0)
        return nn.MaxPool2d(kernelSize, stride, (padAlongHeight, padAlongWidth))(x)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out = self.maxPool2dSame(out2, 2)

        out3 = self.conv3(out)
        out4 = self.conv4(out3)
        out = self.maxPool2dSame(out4, 2)

        out5 = self.conv5(out)
        out6 = self.conv6(out5)
        out = self.maxPool2dSame(out6, 2)

        out7 = self.conv7(out)
        out8 = self.conv8(out7)
        out = self.maxPool2dSame(out8, 2)

        out9 = self.conv9(out)
        out10 = self.conv10(out9)
        out10 = nn.Upsample(scale_factor=2)(out10)
        
        out11 = torch.cat([out8, out10], 1)
        out11 = self.conv11(out11)
        out12 = self.conv12(out11)
        out12 = nn.Upsample(scale_factor=2)(out12)

        out13 = torch.cat([out6, out12], 1)
        out13 = self.conv13(out13)
        out14 = self.conv14(out13)
        out14 = nn.Upsample(scale_factor=2)(out14)
        
        out15 = torch.cat([out4, out14], 1)
        out15 = self.conv15(out15)
        out16 = self.conv16(out15)
        out16 = nn.Upsample(scale_factor=2)(out16)
        
        out17 = torch.cat([out2, out16], 1)
        out17 = self.conv17(out17)
        out18 = self.conv18(out17)
        
        out19 = self.conv19(out18)
        return out19