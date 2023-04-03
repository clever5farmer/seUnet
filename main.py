import torch
import torch.optim as optim
from unet import UNet
import dataLoader as dl
import train, prediction

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(4, 1)
    model = model.to(device=device)
    # read data
    dataset = dl.SegmentationDataset(dataset_dir='../dataset')
    # split dataset
    # train_dataset
    # test_dataset
    # train
    model = train.train(model, device, dataset)
    # test
    pred_img, mask_img = prediction.predict(model, dataset, device)
    # pytorch的unet复现源码大部分上采样用的是反卷积