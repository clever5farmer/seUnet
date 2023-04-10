import torch
import torch.nn as nn
import math
from se_block import SEBlock
from unet import UNet

class SEinFirstUnet(nn.Module):
    def __init__(self, inplanes, planes, ratio=2) -> None:
        super(SEinFirstUnet, self).__init__()
        self.se = SEBlock(inplanes, ratio)
        self.unet = UNet(inplanes, planes)
    
    def forward(self, x):
        out = self.se(x)
        out = self.unet(out)
        return out

class SEBasicBlock(nn.Module):
    def __init__(self, inplanes, features, ratio) -> None:
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, features, 3, padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(features, features, 3, padding='same', bias=False)
        self.batchnorm2 = nn.BatchNorm2d(features)
        self.se = SEBlock(features, ratio)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.relu2(out)
        return out
    
class SEUnet(nn.Module):
    def __init__(self, inplanes, planes, stride=1) -> None:
        super(SEUnet, self).__init__()
        self.encoder1 = SEBasicBlock(inplanes, 64, 2)
        self.encoder2 = SEBasicBlock(64, 128, 16)
        self.encoder3 = SEBasicBlock(128, 256, 16)
        self.encoder4 = SEBasicBlock(256, 512, 16)
        self.bottom = SEBasicBlock(512, 1024, 16)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = SEBasicBlock(1024, 512, 16)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = SEBasicBlock(512, 256, 16)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = SEBasicBlock(256, 128, 16)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = SEBasicBlock(128, 64, 16)
        self.conv19 =  nn.Sequential(
            nn.Conv2d(64, planes, 1, padding='same'),
            nn.Sigmoid())
    
    @staticmethod
    def _maxPool2dSame(x: torch.Tensor, kernelSize, stride = 2):
        _, _, inWidth, inHeight = x.size()
        outWidth = math.ceil(float(inWidth)/float(stride))
        outHeight = math.ceil(float(inHeight)/float(stride))
        padAlongWidth = max((outWidth - 1) * stride + kernelSize - inWidth,0)
        padAlongHeight = max((outHeight - 1) * stride + kernelSize - inHeight,0)
        return nn.MaxPool2d(kernelSize, stride, (padAlongHeight, padAlongWidth))(x)

    def forward(self, x):
        enc1 = self.encoder1(x)
        pool1 = SEUnet._maxPool2dSame(enc1, 2)

        enc2 = self.encoder2(pool1)
        pool2 = SEUnet._maxPool2dSame(enc2, 2)

        enc3 = self.encoder3(pool2)
        pool3 = SEUnet._maxPool2dSame(enc3, 2)

        enc4 = self.encoder4(pool3)
        pool4 = SEUnet._maxPool2dSame(enc4, 2)

        bottom = self.bottom(pool4)

        dec4 = self.upconv4(bottom)
        dec4 = torch.cat((dec4, enc4), 1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), 1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), 1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), 1)
        dec1 = self.decoder1(dec1)
        
        out = self.conv19(dec1)
        return out