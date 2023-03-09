import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from loss import DiceLoss
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


def train(model, device, train_dataloader, epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train_loss = [] # loss of every data in each epoch 
    valid_loss = []
    train_epochs_loss = [] # average loss of each epoch
    valid_epochs_loss = []
    step = 0
    dsc_loss = DiceLoss()
    for epoch in tqdm(range(epochs)):
        model.train()
        train_epoch_loss = []
        for i, data in enumerate(train_dataloader):
            x, y_true = data
            x, y_true = x.to(device), y_true.to(device)
            y_pred = model(x)
            optimizer.zero_grad()
            loss = dsc_loss(y_pred, y_true)
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())
            if i%(len(train_dataloader)//4)==0:
                print("epoch={}/{},{}/{}of train, loss={}".format(
                    epoch, epochs, i, len(train_dataloader),loss.item()))
        train_epochs_loss.append(np.average(train_epoch_loss))

        #=============valid==============
        model.eval()
        valid_epoch_loss = []
        for i, data in enumerate(train_dataloader):
            x, y_true = data
            x, y_true = x.to(device), y_true.to(device)
            y_pred = model(x)
            loss = dsc_loss(y_pred, y_true)
            valid_epoch_loss.append(loss.item())
            valid_loss.append(loss.item())
        valid_epochs_loss.append(np.average(valid_epoch_loss))
