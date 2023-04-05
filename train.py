import copy
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import numpy as np
from tqdm import tqdm
from loss import DiceLoss
from torch.utils.data import DataLoader, random_split
#from unet import UNet

def dsc_per_volume(validation_pred, validation_true, patient_slice_index):
    dsc_list = []
    num_slices = np.bincount([p[0] for p in patient_slice_index])
    index = 0
    for p in range(len(num_slices)):
        y_pred = np.array(validation_pred[index : index + num_slices[p]])
        y_true = np.array(validation_true[index : index + num_slices[p]])
        dsc_list.append(dsc(y_pred, y_true))
        index += num_slices[p]
    return dsc_list


def train(
        model, 
        device, 
        dataset,
        batch_size: int = 1,
        epochs=100):
    summary(model, input_size=(4,256,256))
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    val_percent = 0.33
    val_size = int(len(dataset)*val_percent)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(0)) # 该生成器设置的种子使每次运行该文件都会产生同样的分组
    train_loader = DataLoader(
        train_set, batch_size=batch_size, drop_last=False, num_workers=1
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, drop_last=False, num_workers=1
    )
    # train_epochs_loss = 0 # average loss of each epoch
    step = 0
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    dsc_loss = DiceLoss()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    for epoch in range(1, epochs+1):
    #for epoch in tqdm(range(epochs), total=epochs):
        train_loss = 0 # loss of every data in each epoch 
        model.train()
        with tqdm(total=train_size, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                x, y_true = batch
                x, y_true = x.to(device=device, dtype=torch.float), y_true.to(device=device, dtype=torch.float)
                y_pred = model(x)
                optimizer.zero_grad()
                loss = criterion(y_pred, y_true)
                #cv2.imwrite('/home/luosj/research/test/seUnet/pred.png',y_pred.cpu().detach().numpy())
                #cv2.imwrite('/home/luosj/research/test/seUnet/true.png',y_true.cpu().detach().numpy())
                #print(loss.item())
                #print(np.max(y_true.cpu().detach().numpy()))
                loss.backward()
                optimizer.step()
                train_loss+=loss.item()
                pbar.update(1)
        train_epoch_loss = train_loss/len(train_loader)
        print("\n epoch: {}, train_loss: {:.4f}".format(epoch, train_epoch_loss))

        #=============valid==============
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y_true = batch
                x, y_true = x.to(device, dtype=torch.float), y_true.to(device, dtype=torch.float)
                y_pred = model(x)
                loss = criterion(y_pred, y_true)
                valid_loss+=loss.item()
        valid_epoch_loss = valid_loss/len(val_loader)
        print("\n epoch: {}, val_loss: {:.4f}".format(epoch, valid_epoch_loss))
        if valid_epoch_loss < best_loss:
            best_loss = valid_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_wts)
    return model
