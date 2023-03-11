import os

import cv2
from matplotlib import pyplot as plt
import numpy as np

# 二値化処理(判別分析法)
def threshold_otsu(gray, min_value=0, max_value=255):
    """
    濃淡画像を二値化する．
    ## <Args>
        - gray: 入力画像 
        - min: 二値の小さい方(デフォルト: 0)
        - max: 二値の大きい方(デフォルト: 255)
    
    ## <Returns>
        - gray: 二値画像
    """

    # ヒストグラムの算出
    hist = [np.sum(gray == i) for i in range(256)]

    s_max = (0,-10)

    for th in range(256):
        
        # クラス1とクラス2の画素数を計算
        n1 = sum(hist[:th])
        n2 = sum(hist[th:])
        
        # クラス1とクラス2の画素値の平均を計算
        if n1 == 0 : mu1 = 0
        else : mu1 = sum([i * hist[i] for i in range(0,th)]) / n1   
        if n2 == 0 : mu2 = 0
        else : mu2 = sum([i * hist[i] for i in range(th, 256)]) / n2

        # クラス間分散の分子を計算
        s = n1 * n2 * (mu1 - mu2) ** 2

        # クラス間分散の分子が最大のとき、クラス間分散の分子と閾値を記録
        if s > s_max[1]:
            s_max = (th, s)
    
    # クラス間分散が最大のときの閾値を取得
    t = s_max[0]

    # 算出した閾値で二値化処理
    gray[gray < t] = min_value
    gray[gray >= t] = max_value

    return gray
    

def one_image(input_path, color_type='Gray'):
    """
    入力されたパスの画像を読み出します。(注意: 一枚のみ)
    ### <Args>
        input_path: 画像のパス
        color_type: カラータイプColor or RGB or Gray(デフォルトはGray)
    ### <Returns>
        画像
    """
    if color_type == 'Color':
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    elif color_type == 'RGB':
        _img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
    elif color_type == 'Gray':
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    else:
        print('color_type is wrong!')
    
    height, width = img.shape
    
    img = np.asarray([img], dtype=np.float64)
    img = np.asarray(img, dtype=np.float64).reshape((height, width, 1))
 
    return img


def directory_images(dirpath, imagesize, type_color):
    """
    入力されたディレクトリ内の、画像ファイルのみを読み出します。
    このとき、画像データはnumpyの形式に変更します。(注意: フォルダ内の画像すべて)
    ### <Args>
        dirpath: 画像のパス。
        imagesize: 整形したい画像サイズの大きさ。
        color_type: カラータイプ。ColorまたはGray。
    ### <Returns>
        numpy形式にした画像リストと、
        ファイル名リストを返します。
    """
    imglist = []
    exclude_prefixes = ('__', '.') 

    for root, dirs, files in os.walk(dirpath):
        dirs[:] = [dir for dir in dirs if not dir.startswith(exclude_prefixes)]
        files[:] = [file for file in files if not file.startswith(exclude_prefixes)]

        for fn in sorted(files):
            bn, ext = os.path.splitext(fn)
            if ext not in [".bmp", ".BMP", ".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
                continue

            filename = os.path.join(root, fn)
            
            if type_color == 'Color':
                # カラー画像の場合
                testimage = cv2.imread(filename, cv2.IMREAD_COLOR)
                # testimage = cv2.equalizeHist(testimage)
                # サイズ変更
                height, width = testimage.shape[:2]
                testimage = cv2.resize(testimage, (imagesize, imagesize), interpolation = cv2.INTER_AREA)  #主に縮小するのでINTER_AREA使用
                testimage = np.asarray(testimage, dtype=np.float64)

            elif type_color == 'Gray':
                # グレースケール画像の場合
                testimage = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                # testimage = cv2.equalizeHist(testimage)
                # サイズ変更
                height, width = testimage.shape[:2]
                testimage = cv2.resize(testimage, (imagesize, imagesize), interpolation = cv2.INTER_AREA)  #主に縮小するのでINTER_AREA使用
                # チャンネルの次元がないので1次元追加する
                testimage = np.asarray([testimage], dtype=np.float64)
                testimage = np.asarray(testimage, dtype=np.float64).reshape((imagesize, imagesize, 1))
                # 高さ，幅，チャンネルに入れ替え．data_format="channels_last"を使うとき必要
                # testimage = testimage.transpose(1, 2, 0)
            
            elif type_color == 'Binary_OTSU':
                # グレースケール画像の場合
                testimage = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                # testimage = testimage.astype("uint64")
                testimage = threshold_otsu(testimage)
                # サイズ変更
                height, width = testimage.shape[:2]
                testimage = cv2.resize(testimage, (imagesize, imagesize), interpolation = cv2.INTER_AREA)  #主に縮小するのでINTER_AREA使用
                # チャンネルの次元がないので1次元追加する
                testimage = np.asarray([testimage], dtype=np.float64)
                testimage = np.asarray(testimage, dtype=np.float64).reshape((imagesize, imagesize, 1))
                # 高さ，幅，チャンネルに入れ替え．data_format="channels_last"を使うとき必要
                # testimage = testimage.transpose(1, 2, 0)

            imglist.append(testimage)
    imgsdata = np.asarray(imglist, dtype=np.float32)

    return imgsdata, sorted(files)  # 画像リストとファイル名のリストを返す


def directory_mc_images(dirpath, imagesize, type_color):
    """
    入力されたディレクトリ内の、画像ファイルのみを読み出します。
    このとき、画像データはnumpyの形式に変更します。(注意: フォルダ内の画像すべて)
    ### <Args>
        dirpath: 画像のパス。
        imagesize: 整形したい画像サイズの大きさ。
        color_type: カラータイプ。ColorまたはGray。
    ### <Returns>
        numpy形式にした画像リストと、
        ファイル名リストを返します。
    """
    imglist = []
    exclude_prefixes = ('__', '.') 

    for root, dirs, files in os.walk(dirpath):
        dirs[:] = [dir for dir in dirs if not dir.startswith(exclude_prefixes)]
        files[:] = [file for file in files if not file.startswith(exclude_prefixes)]

        for fn in sorted(files):
            bn, ext = os.path.splitext(fn)
            if ext not in [".npy", '.PNG', '.png']:
                continue

            filename = os.path.join(root, fn)
            
            if type_color == 'Color':
                # カラー画像の場合
                testimage = np.load(filename,allow_pickle=True)
                # testimage = cv2.imread(filename, cv2.IMREAD_COLOR)
                # testimage = cv2.equalizeHist(testimage)
                # サイズ変更
                # height, width = testimage.shape[:2]
                # testimage = cv2.resize(testimage, (imagesize, imagesize), interpolation = cv2.INTER_AREA)  #主に縮小するのでINTER_AREA使用
                testimage = np.asarray(testimage, dtype=np.float64)

            elif type_color == 'Gray':
                # グレースケール画像の場合
                testimage = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                # testimage = cv2.equalizeHist(testimage)
                # サイズ変更
                height, width = testimage.shape[:2]
                testimage = cv2.resize(testimage, (imagesize, imagesize), interpolation = cv2.INTER_AREA)  #主に縮小するのでINTER_AREA使用
                # チャンネルの次元がないので1次元追加する
                testimage = np.asarray([testimage], dtype=np.float64)
                testimage = np.asarray(testimage, dtype=np.float64).reshape((imagesize, imagesize, 1))
                # 高さ，幅，チャンネルに入れ替え．data_format="channels_last"を使うとき必要
                # testimage = testimage.transpose(1, 2, 0)

            imglist.append(testimage)
    imgsdata = np.asarray(imglist, dtype=np.float32)
    
    return imgsdata, sorted(files)  # 画像リストとファイル名のリストを返す


def directory(input_dir, include_ext=[], exclude_ext=[]):
    """
    入力ディレクトリ内の、指定した拡張子のファイルのリストと、
    ファイル名のリストを返します
    ### <Args>
        input_dir: ディレクトリの名前
        include_ext: 読み込みたい拡張子のリスト
        exclude_ext: 除外した拡張子のリスト
        注) 拡張子の前には.を入れる(例: '.png')
    ### <Returns>
        ディレクトリにあるファイル名
    """
    path_list = []
    name_list = []
    for root, dirs, files in os.walk(input_dir):
        for file in sorted(files):
            fn, ext = os.path.splitext(file)

            need_flag = True
            if (len(include_ext) != 0) and (ext not in include_ext):
                need_flag = False
            if (len(exclude_ext) != 0) and (ext in exclude_ext):
                need_flag = False

            if need_flag:
                path = os.path.join(root, file)
                path_list.append(path)
                name_list.append(fn)
            else:
                continue

    return path_list, name_list

def readGroupInfo(dirPath):
    groupList = []
    for root, dirs, files in os.walk(dirPath):
        for file in sorted(files):
            fileName = []
            if not(file.startswith("group")) or not(file.endswith(".csv")):
                continue
            filePath = os.path.join(root, file)
            with open(filePath, "r") as f:
                for line in f.readlines():
                    line = line.strip()
                    fileName.append(line)
            groupList.append(fileName)
    return groupList
