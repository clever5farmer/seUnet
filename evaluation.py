import cv2
import os
import numpy as np
import pandas as pd

def compare_res_lab(image, label):
    '''
    '''
    ysize, xsize = label.shape[:2]
    _ysize, _xsize = image.shape[:2]
    if (ysize != _ysize) or (xsize != _xsize):
        image = cv2.resize(image, (xsize, ysize), interpolation=cv2.INTER_CUBIC)

    # 臒l�ɂ���l��
    threshold = 1
    # th, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    eval_image = np.zeros((ysize, xsize,3), dtype=np.uint8)
    tp = fp = fn = tn = 0.00000001
    for y in range(0,ysize):
        for x in range(0,xsize):
            if label[y][x] == image[y][x]:
                if label[y][x] == 255:
                    tp = tp + 1
                    eval_image[y][x][1] = 255
                else:
                    tn = tn + 1
            else:
                if label[y][x] == 255:
                    fn = fn + 1
                    eval_image[y][x][0] = 255
                else:
                    fp = fp + 1
                    eval_image[y][x][2] = 255

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fMesure = 2 * precision * recall / (precision + recall)
    return precision, recall, fMesure, eval_image


def overlay_eval(image, oriimage, label):
    '''
    '''
    ysize, xsize = label.shape[:2]
    _ysize, _xsize = image.shape[:2]
    if (ysize != _ysize) or (xsize != _xsize):
        image = cv2.resize(image, (xsize, ysize), interpolation=cv2.INTER_CUBIC)

    # 臒l�ɂ���l��
    threshold = 1
    # th, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    eval_image = np.zeros((ysize, xsize,3), dtype=np.uint8)

    for y in range(0,ysize):
        for x in range(0,xsize):
            if label[y][x] == image[y][x]:
                if label[y][x] == 255:
                    eval_image[y][x][1] = 255
                else:
                    eval_image[y][x][0] = oriimage[y][x]
                    eval_image[y][x][1] = oriimage[y][x]
                    eval_image[y][x][2] = oriimage[y][x]
            else:
                if label[y][x] == 255:
                    eval_image[y][x][0] = 255
                else:
                    eval_image[y][x][2] = 255

    return eval_image


# ��l������(���ʕ��͖@)
def threshold_otsu(image, min_value=0, max_value=255):
    '''
    '''
    if len(image.shape) == 3:
        height, width, _ = image.shape
        image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    else:
        height, width = image.shape

    # �q�X�g�O�����̎Z�o
    hist = [np.sum(image == i) for i in range(256)]

    s_max = (0,-10)

    for th in range(256):
        
        # �N���X1�ƃN���X2�̉�f�����v�Z
        n1 = sum(hist[:th])
        n2 = sum(hist[th:])
        
        # �N���X1�ƃN���X2�̉�f�l�̕��ς��v�Z
        if n1 == 0 : mu1 = 0
        else : mu1 = sum([i * hist[i] for i in range(0,th)]) / n1   
        if n2 == 0 : mu2 = 0
        else : mu2 = sum([i * hist[i] for i in range(th, 256)]) / n2

        # �N���X�ԕ��U�̕��q���v�Z
        s = n1 * n2 * (mu1 - mu2) ** 2

        # �N���X�ԕ��U�̕��q���ő�̂Ƃ��A�N���X�ԕ��U�̕��q��臒l���L�^
        if s > s_max[1]:
            s_max = (th, s)
    
    # �N���X�ԕ��U���ő�̂Ƃ���臒l���擾
    t = s_max[0]

    # �Z�o����臒l�œ�l������
    image[image < t] = min_value
    image[image >= t] = max_value

    return image

def save_eval_result(predict_image, gt_image, test_filenames, resDir, mask_image=None, overlay_on=False, ori_image=None):
    '''
    '''
    # �㏈����̉摜�t�H���_
    resBinDir = os.path.join(resDir,'postProc-predict/')
    os.makedirs(resBinDir, exist_ok=True)

    # �]���摜�̃t�H���_
    evalDir = os.path.join(resDir, 'evaluation/')
    os.makedirs(evalDir, exist_ok=True)

    if overlay_on:
        # �]���摜�̃t�H���_
        overlayDir = os.path.join(resDir, 'overlay/')
        os.makedirs(overlayDir, exist_ok=True)

    # �]�����ʂ�CSV
    evalTotalCsv = os.path.join(resDir, 'evaluation.csv')

    # ����SACsv�͂��łɍ���Ă���t�@�C���Ȃ炱���͎��s����Ȃ�
    if not os.path.exists(evalTotalCsv):
        eva_total_df = pd.DataFrame(index=['case'], columns=['precision', 'recall', 'F-measure'])
        eva_total_df.to_csv(evalTotalCsv)

    eva_total_df = pd.read_csv(evalTotalCsv, index_col=0)
    eva_total_df.to_csv(evalTotalCsv)

    binImgs = [threshold_otsu(img) for img in predict_image]

    # precision, recall, F�l�̕��ς����߂�
    meanPrecision = 0.0
    meanRecall = 0.0
    meanFmeasure = 0.0

    # �]���v�Z
    for i in range(0, len(binImgs)):
        # �\�����ꂽ�m���摜�ɂ��Č㏈�����{��
        # �����ł͒P�ɓ�l������
        binImg = binImgs[i]
        cv2.imwrite(resBinDir + test_filenames[i], binImg)

        # �����摜�̓ǂݍ���
        # gt_img_path = gtDir + test_filenames[i]
        gtImg = gt_image[i]

        # �}�X�N�摜������ꍇ�ɂ̓}�X�N�O�̌��ʂ����O����
        if (mask_image is not None):
            ysize, xsize = binImg.shape[:2]
            for y in range(0,ysize):
                for x in range(0,xsize):
                    if mask_image[i][y][x] == 0:
                        binImg[y][x] = 0
        

        # TP/FP/FN�̐F����
        precision, recall, Fmeasure, evalImg = compare_res_lab(binImg, gtImg)
        cv2.imwrite(evalDir + test_filenames[i], evalImg)

        if overlay_on:
            oriImg = ori_image[i]
            overlayImg = overlay_eval(binImg, oriImg, gtImg)
            cv2.imwrite(overlayDir + test_filenames[i], overlayImg)

        # CSV�Ɍ��ʂ̏�������
        case_name = test_filenames[i].split('.')[0]
        eva_total_df.loc[case_name] = [precision, recall, Fmeasure]

        # ���όv�Z�r��
        meanPrecision += precision
        meanRecall += recall
        meanFmeasure += Fmeasure

    # ���όv�Z
    meanPrecision /= len(binImgs)
    meanRecall /= len(binImgs)  
    meanFmeasure /= len(binImgs)

    # CSV�ɕ��ς̌��ʂ���������
    eva_total_df.loc['Average'] = [meanPrecision, meanRecall, meanFmeasure]
    eva_total_df.to_csv(evalTotalCsv)

    return meanPrecision, meanRecall, meanFmeasure

def SummerizeResults(KConst, resultDir):
    total_res_df_n = pd.DataFrame()
    #total_res_df_n.to_csv(os.path.join(resultDir +'total_average_evaluation.csv'),index=0)
    print(total_res_df_n)
    for j in range(KConst):
        unet_res_df = pd.DataFrame([pd.read_csv(os.path.join(resultDir, '{0}'.format(j), 'unet', 'testevaluation.csv'), index_col=0).loc['Average']])
        unet_res_df.columns = ['precision(unet)', 'recall(unet)', 'F-measure(unet)']
        pix_res_df =  pd.DataFrame([pd.read_csv(os.path.join(resultDir, '{0}'.format(j), '1x1unet', 'testevaluation.csv'), index_col=0).loc['Average']])
        pix_res_df.columns = ['precision(P)', 'recall(P)', 'F-measure(P)']

        single_res = pd.concat([unet_res_df, pix_res_df],axis=1)
        total_res_df_n = pd.concat([total_res_df_n, single_res],ignore_index=True)
        print(total_res_df_n)

    #total_res_df_n['precision(unet)'] /= float(KConst)
    #total_res_df_n['recall(unet)']    /= float(KConst)
    #total_res_df_n['F-measure(unet)'] /= float(KConst) 

    #total_res_df_n['precision(P)'] /= float(KConst)
    #total_res_df_n['recall(P)']    /= float(KConst)
    #total_res_df_n['F-measure(P)'] /= float(KConst)

    # 平均を求める
    total_res_df_n.loc['Average'] = total_res_df_n.mean()

    total_res_df_n.to_csv(os.path.join(resultDir,'total_average_evaluation.csv'))