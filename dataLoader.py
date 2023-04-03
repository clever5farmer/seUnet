import os
import torch
import numpy as np
import preprocess as prep
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from Rfund.InputFeature import InputFeature

def data_loader(dataset_dir, flag='train', batch_size=1):
    dataset = SegmentationDataset(dataset_dir, flag)
    loader = DataLoader(
        dataset, batch_size=batch_size, drop_last=False, num_workers=1
    )
    return loader

class SegmentationDataset(Dataset):
    def __init__(self, dataset_dir) -> None:

        originalImgDir = os.path.join(dataset_dir, 'original/image')
        featureRootDir = os.path.join(dataset_dir, 'feature/Original')
        labelRootDir = os.path.join(dataset_dir, 'original/label/')
        print("reading images...")
        imgList, fileNameList = prep.readImages(originalImgDir)
        labelImgList, _ = prep.readImages(labelRootDir)
        dataSize = len(imgList)
        imgSize = np.shape(imgList)[-2:]
        multiChannelImages = np.zeros((dataSize,len(InputFeature),imgSize[0],imgSize[1]))
        i=0
        for inputFeature in InputFeature:
            featureVal = inputFeature.value
            imgPath = os.path.join(featureRootDir, featureVal)
            #print("imgPath", imgPath)
            imgList, fileNameList = prep.readImages(imgPath)
            prep.stackChannelImages(multiChannelImages, imgList, i)
            i+=1
        self.transform = transforms.Compose([
            transforms.ToTensor() # 0-255 -> 0-1, dimensions (H, W, C) -> (C, H, W)
        ])
        self.multiChannelImages = multiChannelImages
        self.rawImages = imgList
        self.labelImages = labelImgList
        self.labelImages = self.labelImages[..., np.newaxis]
        self.fileNames = fileNameList
        '''
        self.multiChannelImages = multiChannelImages / np.double(255) # dimensions (N, C, H, W)
        self.rawImages = imgList / np.double(255)
        self.labelImages = labelImgList / np.double(255)
        self.labelImages = self.labelImages[..., np.newaxis]
        self.labelImages = torch.reshape(self.labelImages, [0,3,1,2]) # dimensions (N, H, W, C) -> (N, C, H, W)
        self.fileNames = fileNameList
        '''

    def __len__(self):
        return len(self.fileNames)
    
    def __getitem__(self, index):
        image = self.multiChannelImages[index] # dimensions (C, H, W)
        label = self.labelImages[index] # dimensions (C, H, W)
        #image_tensor = torch.from_numpy(image.astype(np.float32))
        #label_tensor = torch.from_numpy(label.astype(np.float32))
        image_tensor = self.transform(image)
        label_tensor = self.transform(label)
        return [image_tensor, label_tensor]
        

