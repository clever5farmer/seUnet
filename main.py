import torch
import torch.optim as optim
from unet import UNet
import dataLoader

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(4, 1)
    model = model.to(device=device)
    # read data
    loader = dataLoader(dataset_dir='../dataset')
    # split dataset
    # train
    # test
    # pytorch的unet复现源码大部分上采样用的是反卷积