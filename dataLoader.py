import os
import torch
import numpy as np
import preprocess as prep
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from Rfund.InputFeature import InputFeature

def data_loader(dataset_dir, flag='train', batch_size=1):
    dataset = SegmentationDataset(dataset_dir, flag)
    loader = DataLoader(
        dataset, batch_size=batch_size, drop_last=False, num_workers=1
    )
    return loader

class SegmentationDataset(Dataset):
    def __init__(self, dataset_dir, flag='train') -> None:
        assert flag in ['train','test','valid']

        originalImgDir = os.path.join(dataset_dir, 'original/image')
        featureRootDir = os.path.join(dataset_dir, 'feature/Original')
        labelRootDir = os.path.join(dataset_dir, 'original/label/')
        print("reading {} images...".format(flag))
        imgList, fileNameList = prep.readImages(originalImgDir)
        labelImgList, _ = prep.readImages(labelRootDir)
        dataSize = len(imgList)
        imgSize = np.shape(imgList)[-2:]
        multiChannelImages = np.zeros((dataSize,imgSize[0],imgSize[1],len(InputFeature)))
        i=0
        for inputFeature in InputFeature:
            featureVal = inputFeature.value
            imgPath = os.path.join(featureRootDir, featureVal)
            #print("imgPath", imgPath)
            imgList, fileNameList = prep.readImages(imgPath)
            prep.stackChannelImages(multiChannelImages, imgList, i)
            i+=1
        self.multiChannelImages = multiChannelImages / np.double(255) # dimensions (H, W, C)
        self.rawImages = imgList / np.double(255)
        self.labelImages = labelImgList / np.double(255)
        self.labelImages = self.labelImages[..., np.newaxis] # dimensions (H, W, C) 
        self.fileNames = fileNameList

    def __len__(self):
        return len(self.fileNames)
    
    def __getitem__(self, index):
        image = self.multiChannelImages[index] # dimensions (H, W, C)
        label = self.labelImages[index] # dimensions (H, W, C)
        image = image.transpose(2, 0, 1) # dimensions (C, H, W)
        label = label.transpose(2, 0, 1) # dimensions (C, H, W)
        image_tensor = torch.from_numpy(image.astype(np.float32))
        label_tensor = torch.from_numpy(label.astype(np.float32))
        return image_tensor, label_tensor
        

