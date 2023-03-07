import torch
import torch.optim as optim
from tqdm import tqdm
#from unet import UNet

def train(model, device, epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_train = []
    loss_valid = []
    step = 0
    for epoch in tqdm(range(epochs)):
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            validation_pred = []
            validation_true = []
            for i, data in enumerate(data[phase]):
                if phase == "train":
                    step +=1
                x, y_true = data
                x, y_true = x.to(device), y_true.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    y_pred = model(x)
                    loss = 
            
