import cv2
import numpy as np

"""
これは使わない
"""

def mfeatureImages(marray, LEARN_CASES, TEST_CASES, lineDir, datasetDir, rootDir, testmode=False):
    """
    mチャンネルにそれぞれの特徴画像(グレースケール)を読み込んで、
    一枚のmチャンネルの画像に変換する
    ### <Args>
        marray: m個の特徴画像の名前(String)
        LEARN_CASES: 学習のグループ(String)
        TEST_CASES: テストのグループ(String)
        lineDir: 椎間板のラベル画像があるディレクトリ
        datasetDir: フィルタをかけた椎間板画像があるディレクトリ
        rootDir: 作成された画像を保存するディレクトリ
    ### <Returns>
        なし
    """
    m = len(marray)
    multichannel_img_list = []
    for i in range(0,m) :
        multichannel_img_list.append(datasetDir + 'feature_png/' + marray[i] + '/')

    ftTrainDir = rootDir + 'org_train/no_class/'
    labelTrainDir = rootDir + 'label_train/no_class/'
    ftTestDir = rootDir + 'org_test/no_class/'
    labelTestDir = rootDir + 'label_test/no_class/'

    if testmode:
        ftTestDir = rootDir + 'org_test2/no_class/'
        labelTestDir = rootDir + 'label_test2/no_class/'
    else:
        ftTestDir = rootDir + 'org_test/no_class/'
        labelTestDir = rootDir + 'label_test/no_class/'    

    for learnName in LEARN_CASES:
        img = cv2.imread(multichannel_img_list[0] + learnName + '.png', cv2.IMREAD_GRAYSCALE)
        for i in range(1,m) :
            _img = cv2.imread(multichannel_img_list[i] + learnName + '.png', cv2.IMREAD_GRAYSCALE)
            img = np.dstack([img, _img])
        np.save(ftTrainDir + learnName, img)
        lineImg = cv2.imread(lineDir + learnName + '.png', cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(labelTrainDir + learnName + '.png', lineImg)

    for testName in TEST_CASES:
        img = cv2.imread(multichannel_img_list[0] + testName + '.png', cv2.IMREAD_GRAYSCALE)
        for i in range(1,m) :
            _img = cv2.imread(multichannel_img_list[i] + testName + '.png', cv2.IMREAD_GRAYSCALE)
            img = np.dstack([img, _img])
        np.save(ftTestDir + testName, img)
        lineImg = cv2.imread(lineDir + testName + '.png', cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(labelTestDir + testName + '.png', lineImg)


# def featureImages(RGB, LEARN_CASES, TEST_CASES, lineDir, datasetDir, rootDir):
def featureImages(RGB, LEARN_CASES, TEST_CASES, lineDir, datasetDir, rootDir, testmode=False):
    """
    RGBそれぞれの特徴画像(グレースケール画像)を読み込んで、
    一枚のカラー画像に変換する
    ### <Args>
        RGB: 3つの特徴の名前(String)
        LEARN_CASES: 学習のグループ(String)
        TEST_CASES: テストのグループ(String)
        lineDir: 椎間板のラベル画像があるディレクトリ
        datasetDir: フィルタをかけた椎間板画像があるディレクトリ
        rootDir: 作成された画像を保存するディレクトリ
    ### <Returns>
        なし
    """
    # RGBチャンネルのそれぞれ特徴画像を入れる。
    Rdir = datasetDir + 'feature_png/' + RGB[0] + '/'
    Gdir = datasetDir + 'feature_png/' + RGB[1] + '/'
    Bdir = datasetDir + 'feature_png/' + RGB[2] + '/'
    ftTrainDir = rootDir + 'org_train/no_class/'
    labelTrainDir = rootDir + 'label_train/no_class/'

    if testmode:
        ftTestDir = rootDir + 'org_test2/no_class/'
        labelTestDir = rootDir + 'label_test2/no_class/'
    else:
        ftTestDir = rootDir + 'org_test/no_class/'
        labelTestDir = rootDir + 'label_test/no_class/'
    
    for learnName in LEARN_CASES:
        ftR = cv2.imread(Rdir + learnName + '.png', cv2.IMREAD_GRAYSCALE)
        ftR = np.asarray([ftR], dtype=np.float64)
        height, width = ftR.shape[:2]
        ftR = np.asarray(ftR, dtype=np.float64).reshape((width, width,1))
        ftG = cv2.imread(Gdir + learnName + '.png', cv2.IMREAD_GRAYSCALE)
        ftG = np.asarray([ftG], dtype=np.float64)
        height, width = ftR.shape[:2]
        ftG = np.asarray(ftG, dtype=np.float64).reshape((width, width,1))
        ftB = cv2.imread(Bdir + learnName + '.png', cv2.IMREAD_GRAYSCALE)
        ftB = np.asarray([ftB], dtype=np.float64)
        height, width = ftR.shape[:2]
        ftB = np.asarray(ftB, dtype=np.float64).reshape((width, width,1))
        ftImg = np.dstack((np.dstack((ftB, ftG)), ftR))
        cv2.imwrite(ftTrainDir + learnName + '.png', ftImg)
        lineImg = cv2.imread(lineDir + learnName + '.png', cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(labelTrainDir + learnName + '.png', lineImg)

    for testName in TEST_CASES:
        ftR = cv2.imread(Rdir + testName + '.png', cv2.IMREAD_GRAYSCALE)
        ftG = cv2.imread(Gdir + testName + '.png', cv2.IMREAD_GRAYSCALE)
        ftB = cv2.imread(Bdir + testName + '.png', cv2.IMREAD_GRAYSCALE)
        ftImg = np.dstack((np.dstack((ftB, ftG)), ftR))
        cv2.imwrite(ftTestDir + testName + '.png', ftImg)
        lineImg = cv2.imread(lineDir + testName + '.png', cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(labelTestDir + testName + '.png', lineImg)
