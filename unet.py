import torch
import torch.nn as nn
import math
from collections import OrderedDict

class UNet(nn.Module):
    def __init__(self, inplanes, planes, stride=1) -> None:
        super(UNet, self).__init__()
        self.encoder1 = UNet._convBlock(inplanes, 64, name='enc1')
        self.encoder2 = UNet._convBlock(64, 128, name='enc2')
        self.encoder3 = UNet._convBlock(128, 256, name='enc3')
        self.encoder4 = UNet._convBlock(256, 512, name='enc4')
        self.bottom = UNet._convBlock(512, 1024, name='bottom')
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = UNet._convBlock(1024, 512, name='dec4')
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = UNet._convBlock(512, 256, name='dec3')
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = UNet._convBlock(256, 128, name='dec2')
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = UNet._convBlock(128, 64, name='dec1')
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

    @staticmethod
    def _convBlock(inplanes, features, name):
        return nn.Sequential(
            OrderedDict([
                    (name + 'conv1', nn.Conv2d(inplanes, features, 3, padding='same', bias=False)),
                    (name + 'batchnorm1', nn.BatchNorm2d(features)),
                    (name + 'relu1', nn.ReLU(inplace=True)),
                    (name + 'conv2', nn.Conv2d(features, features, 3, padding='same', bias=False)),
                    (name + 'batchnorm2', nn.BatchNorm2d(features)),
                    (name + 'relu2', nn.ReLU(inplace=True))
                ])
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        pool1 = UNet._maxPool2dSame(enc1, 2)

        enc2 = self.encoder2(pool1)
        pool2 = UNet._maxPool2dSame(enc2, 2)

        enc3 = self.encoder3(pool2)
        pool3 = UNet._maxPool2dSame(enc3, 2)

        enc4 = self.encoder4(pool3)
        pool4 = UNet._maxPool2dSame(enc4, 2)

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

#net = UNet(4,1)
#print(list(net.parameters()))