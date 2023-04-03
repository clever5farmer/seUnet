import torch
from torch.utils.data import DataLoader
import numpy as np

def predict(
        model,
        dataset,
        device,
        img_size = 256,
        out_threshold=0.5):
    model.eval()
    test_loader = DataLoader(
        dataset, batch_size=1, drop_last=False, num_workers=1
    )
    activator = torch.nn.Sigmoid()
    pred_images = np.zeros((len(dataset), img_size, img_size))
    mask_images = np.zeros((len(dataset), img_size, img_size))
    with torch.no_grad():
            for i, batch in enumerate(test_loader):
                x, _ = batch
                x= x.to(device)
                y_pred = model(x).cpu()
                mask = activator(y_pred) > out_threshold
                pred_images[i,:,:]=y_pred.numpy()
                mask_images[i,:,:]=mask.numpy()
    return pred_images, mask_images
