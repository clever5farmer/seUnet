import os
import sys
import time
import dataLoader as dl
import prediction
import torch
import torch.optim as optim
import numpy as np
import train
from unet import UNet
from se_unet import SEUnet
from evaluation import save_eval_result, SummerizeResults

KCONST = 3
NEGATIVENUM = 39
POSITIVENUM = 19
'''
randOrder = [3,4,9,11,17,20,21,22,23,25,27,28,37,41,49,50,52,54,57,
            1,2,5,6,8,10,13,18,26,33,34,35,38,39,44,45,47,48,53,
            0,7,12,14,15,16,19,24,29,30,31,32,36,40,42,43,46,51,55,56]
negativeCases = [3, 4, 9, 11, 17, 20, 21, 22, 23, 25, 27, 28, 37, 
                1, 2, 5, 6, 8, 10, 13, 18, 26, 33, 34, 35, 0, 7,
                38, 12, 14, 15, 16, 19, 24, 29, 30, 31, 32, 36]
positiveCases = [41, 49, 50, 52, 54, 57, 39, 44, 45, 47, 
                48, 53, 40, 42, 43, 46, 51, 55, 56]
'''
negSliceList = []
posSliceList = []
dateTime = time.strftime("%Y%m%d_%H_%M", time.localtime())
resultDir = os.path.join('../result/seUnet', dateTime)
def get_set_slice(KConst, NSize, PSize, Round):
    if negSliceList != []:
        return negSliceList[Round], posSliceList[Round]
    negID, posID = 0, 0
    for i in range(KConst):
        foldSize = [NSize//KConst, PSize//KConst]
        remainder = [NSize%KConst, PSize%KConst]
        if (i < remainder[0]):
            foldSize[0]+=1
        if (i < remainder[1]):
            foldSize[1]+=1
        oldNegID, oldPosID = negID, posID
        negID, posID = negID+foldSize[0], posID+foldSize[1]
        negSliceList.append([oldNegID, negID])
        posSliceList.append([oldPosID, posID])
    return negSliceList[Round], posSliceList[Round]

def train_by_model(model, model_name, round):
    image_set = dl.ImgageSet(dataset_dir='../dataset')
    test_slice = get_set_slice(KCONST, NEGATIVENUM, POSITIVENUM, round)
    train_dataset = dl.SegmentationDataset(image_set, test_slice, flag='train')
    test_dataset = dl.SegmentationDataset(image_set, test_slice, flag='test')
    model = train.train(model, device, train_dataset)
    # test
    pred_img, mask_img = prediction.predict(model, test_dataset, device)
    label_img = np.concatenate((image_set.labelImages[slice(test_slice[0][0], test_slice[0][1])], 
                                image_set.labelImages[slice(test_slice[1][0], test_slice[1][1])]),
                                axis=0)
    fileName_set = image_set.fileNames[slice(test_slice[0][0], test_slice[0][1])]+image_set.fileNames[slice(test_slice[1][0], test_slice[1][1])]
    test_raw_img = np.concatenate((image_set.rawImages[slice(test_slice[0][0], test_slice[0][1])], 
                                image_set.rawImages[slice(test_slice[1][0], test_slice[1][1])]),
                                axis=0)
    subResDir = os.path.join(resultDir, 'iteration '+str(round), model_name)
    meanPrec, meanReca, meanFmea = save_eval_result(predict_image=pred_img*255.0,
                                                    gt_image=label_img,
                                                    test_filenames=fileName_set,
                                                    resDir=subResDir,
                                                    mask_image=mask_img,
                                                    overlay_on=True,
                                                    ori_image=test_raw_img)

if __name__ == '__main__':
    os.makedirs(resultDir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # read data
    #dataset = dl.SegmentationDataset(dataset_dir='../dataset')
    # split dataset
    for i in range(KCONST):
        # train_dataset
        #np.set_printoptions(threshold=sys.maxsize)
        # test_dataset
        # train
        # print((image_set.labelImages[0,:,:,0]))
        # model = UNet(4, 1)
        '''
        model = SEUnet(4, 1)
        model = model.to(device=device)
        train_by_model(model, 'SEUnet', i)
        '''
        model = UNet(4, 1)
        model = model.to(device=device)
        train_by_model(model, 'Unet', i)

    # pytorch的unet复现源码大部分上采样用的是反卷积