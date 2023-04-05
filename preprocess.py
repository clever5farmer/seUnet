import numpy as np
import imgaug.augmenters as iaa
import cv2
import os
from Rfund.InputFeature import InputFeature
from Rfund.Augmentation import Augmentation
import Rfund.Filtering as Filtering

def readImages(dirPath):
    imgList = []
    fileNameList = []
    '''
    for root, dirs, fs in os.walk(dirPath):
        for dir in dirs:
            key = os.path.basename(dir)
            imgsOfSingleAug = []
            for i in range(len(groupInfo)):
                imgsOfSingleGroupSingleAug = []
                for imgName in groupInfo[i]:
                    imgPath = os.path.join(root, dir, imgName+".png")
                    print(imgPath)
                    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
                    imgsOfSingleGroupSingleAug.append(img)
                imgsOfSingleAug.append(imgsOfSingleGroupSingleAug)
            imgDict[key] = imgsOfSingleAug
    return imgDict
            
    '''
    for root, dirs, fs in os.walk(dirPath):
        for fn in sorted(fs):
            _, ext = os.path.splitext(fn)
            if ext not in [".bmp", ".BMP", ".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
                continue
            fileName = fn
            if (os.path.basename(os.path.dirname(root)) in Augmentation._value2member_map_):
                fileName = os.path.basename(os.path.dirname(root)) + '_' + fileName
            fileNameList.append(fileName)
            filePath = os.path.join(root, fn)
            img = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
            imgList.append(img)
    return np.array(imgList), fileNameList

def readImagesByAugType(groupInfo, dirPath):
    imgDict = {}
    dirs = os.listdir(dirPath)
    for dir in dirs:
        key = os.path.basename(dir)
        imgsOfSingleAug = []
        for i in range(len(groupInfo)):
            imgsOfSingleGroupSingleAug = []
            for imgName in groupInfo[i]:
                imgPath = os.path.join(dirPath, dir, imgName+".png")
                print(imgPath)
                img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
                imgsOfSingleGroupSingleAug.append(img)
            imgsOfSingleAug.append(imgsOfSingleGroupSingleAug)
        imgDict[key] = imgsOfSingleAug
    return imgDict


def saveImages(imgList, fileNameList, dirPath):
    os.makedirs(dirPath, exist_ok=True)
    for i, fn in enumerate(fileNameList):
        filename = os.path.join(dirPath, fn)
        testimage = imgList[i]
        cv2.imwrite(filename, testimage)

def createFeatureImages(imgList, featureDir, fileNameList, augMode = 'Original'):
    """
    create feature images
    ## <Args>
    """ 
    for inputFeature in InputFeature:
        groupFeatureDir = os.path.join(featureDir, augMode, inputFeature.value)
        if os.path.exists(groupFeatureDir):
            continue
        print(groupFeatureDir)
        os.makedirs(groupFeatureDir, exist_ok=True)
        featureImgList = Filtering.list_images(imgList, inputFeature, nml_on=True)
        saveImages(featureImgList, fileNameList, groupFeatureDir)
        # WriteFile.save_images(groupFeatureDir, imageNameList, featureImgList)

def stackChannelImages(multiChannelImages, imgArr, index):
    multiChannelImages[:,:,:,index] = imgArr

def initializeMultiChannelImgs(featureDir):
    multiChannelImgs = []
    for inputFeature in InputFeature:
        for augType in Augmentation:
            imgDir = os.path.join(featureDir, inputFeature, augType)
            imageList, _ = readImages(imgDir)
            multiChannelImgs = stackChannelImages(multiChannelImgs, imageList)
    return multiChannelImgs

def imageAugmentation(imgList, labelImgList, augType):
    if augType == Augmentation.ORIGINAL:
        return imgList, labelImgList
    #if augType == Augmentation.AFFINE:
    #    seq = iaa.Sequential([iaa.Affine(shear=(-16,16))])
    #elif augType == Augmentation.SCALEX:
    #    seq = iaa.Sequential([iaa.ScaleX((0.5, 1.5))])
    #elif augType == Augmentation.SCALEY:
    #    seq = iaa.Sequential([iaa.ScaleY((0.5,1.5))])
    #if augType == Augmentation.TRANSLATION20:
    #    seq = iaa.Sequential([iaa.TranslateX(px=(20,20))])
    #elif augType == Augmentation.TRANSLATION40:
    #    seq = iaa.Sequential([iaa.TranslateX(px=(40,40))])
    if augType == Augmentation.CUTOUT:
        seq = iaa.Sequential([iaa.Cutout(nb_iterations=1, size=0.2, squared=False)])
    if augType == Augmentation.CUTOUT2:
        seq = iaa.Sequential([iaa.Cutout(nb_iterations=2, size=0.25, squared=False)])
    imagesAug, segmapsAug = seq(images=imgList, segmentation_maps=labelImgList)
    return imagesAug, segmapsAug

'''
originalImgDir = '/home/luosj/research/gunjiDocuments/dataset/without_black_area_crop/image/'
featureRootDir = '/home/luosj/research/test/unet/dataset/feature'
labelRootDir = '/home/luosj/research/gunjiDocuments/dataset/without_black_area_crop/label/'
AugImgDir = '/home/luosj/research/test/unet/augmentation/images'
AugLabelDir = '/home/luosj/research/test/unet/augmentation/labels'

os.makedirs(AugImgDir, exist_ok=True)
os.makedirs(AugLabelDir, exist_ok=True)
imgList, fileNameList = readImages(originalImgDir)
print("jdkjfldsj", np.shape(imgList))
labelImgList, _ = readImages(labelRootDir)
labelImgList = np.expand_dims(labelImgList, axis=-1)
#imgPath = os.path.join(originalImgDir, "krhaa001_n_N1_from3072.png")
#labelPath = os.path.join(labelRootDir, "krhaa001_n_N1_from3072.png")
#img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
for augment in Augmentation:
    #if os.path.exists(os.path.join(AugImgDir, augment)):
        #continue
    print(augment.value)
    imagesAug, segmapsAug = imageAugmentation(imgList, labelImgList, augment)
    print(np.shape(imagesAug))
    saveImages(imagesAug, fileNameList, os.path.join(AugImgDir, augment.value))
    saveImages(segmapsAug, fileNameList, os.path.join(AugLabelDir, augment.value))
    createFeatureImages(imagesAug, featureRootDir, fileNameList, augment.value)
allimages, _ = readImages(featureRootDir)
print(np.shape(allimages))
'''