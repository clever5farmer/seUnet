import torch
from unet import UNet

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(4, 1)
    model = model.to(device=device)
    # read data
    # split dataset
    # train
    # test
    # pytorch的unet复现源码大部分上采样用的是反卷积