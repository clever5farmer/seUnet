from enum import Enum

import cv2
import numpy as np
import math

class FilterKernel: 
    """
    フィルタの行列．ここにはソーベル(SBL)，ラプラシアン(LPL)，エンボス(EMB)がある．
    - ソーベル規則: SBL[XorY][3or5][3or5]，例) SBLX33はX方向に3×3
    - ラプラシアン規則: LPL[4or8], LPL4は4近傍，LPL8は8近傍
    - エンボス規則: EMB[33or35or53], 例)EMB33は3×3
    """
    SBLX33 = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])    
    SBLY33 = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])
    SBLX35 = np.array([[-1, 0, 0, 0, 1],
                   [-2, 0, 0, 0, 2],
                   [-1, 0, 0 ,0, 1]])
    SBLY53 = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [1, 2, 1]])
    SBLX55 = np.array([[-1, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0],
                   [-2, 0, 0, 0, 2],
                   [0, 0, 0, 0, 0],
                   [-1, 0, 0 ,0, 1]])
    SBLY55 = np.array([[-1, 0, -2, 0, -1],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [1, 0, 2, 0, 1]])
    LPL4 = np.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]])
    LPL8 = np.array([[1, 1, 1],
                   [1, -8, 1],
                   [1, 1, 1]]) 
    EMB33 = np.array([[-2, -1, 0],
                   [-1, 1, 1],
                   [0, 1, 2]])  
    EMB35 = np.array([[-2, 0, -1, 0, 0],
                   [-1, 0, 1, 0, 1],
                   [0, 0, 1 ,0, 2]])
    EMB53 = np.array([[-2, -1, 0],
                   [0, 0, 0],
                   [-1, 1, 1],
                   [0, 0, 0],
                   [0, 2, 1]])

class ImgStruct_Name(Enum):
    """
    カーネルの構造の形．
    - RECT: 矩形
    - CIRCLE: 円形
    - CROSS: 十字型(現時点使用していない)
    """
    RECT = 0
    CIRCLE = 1
    CROSS = 2

class ImgStruct_Set:
    """
    カーネルの座標集合
    ## <使用例>
    とある大きさ(img_h,img_w)の二次元画像から，
    座標(x,y)中心，ksizeのkernel_shapeに含まれる座標集合を取得したいとき，
    ### ImgStruct_Set(x,y,img_h,img_w).get(kernel_shape, ksize)
    """
    # コンストラクタ
    def __init__(self, x, y, img_size):
        self.x = x
        self.y = y
        self.img_size = img_size
    
    # (x,y)中心のカーネル内にある座標集合
    def get(self, str_name, ksize=1):
        kernel_set = {(self.x, self.y)}
        height, width = self.img_size
        # ksize×ksizeの矩形カーネルに当てはまる座標
        if str_name == ImgStruct_Name.RECT:
            for i in range(-ksize, ksize+1):
                if 0 <= (self.x + i) and (self.x + i) < width:
                    for j in range(-ksize, ksize+1):
                        if 0 <= (self.y + j) and (self.y + j) < height:
                            kernel_set.add((self.x+i, self.y+j))
        # 半径ksizeの円形カーネルに当てはまる座標
        elif str_name == ImgStruct_Name.CIRCLE:
            for i in range(-ksize, ksize+1):
                if 0 <= (self.x + i) and (self.x + i) < width:
                    for j in range(-ksize, ksize+1):
                        if 0 <= (self.y + j) and (self.y + j) < height:
                            if i*i + j*j <=  ksize*ksize:
                                kernel_set.add((self.x+i, self.y+j))
        # 半径ksizeの十字型カーネルに当てはまる座標(デフォルト:４近傍)
        elif str_name == ImgStruct_Name.CROSS:
            for i in range(-ksize, ksize+1):
                if 0 <= (self.x + i) and (self.x + i) < width:
                    kernel_set.add((self.x+i, self.y))
            for j in range(-ksize, ksize+1):
                if 0 <= (self.y + j) and (self.y + j) < height:
                    kernel_set.add((self.x, self.y+j))
        
        return kernel_set

def gray(image):
    """
    型変換.あとで要確認
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    return image

def normalization(image, s):
    """
    画像を[-s*σ, s*σ]から[0,255]に正規化します．
    ## <Args>
        - image: 入力画像 
        - s: 取る領域(1 or 2 or 3)
    
    ## <Returns>
        - f_img: 正規化した画像
    """
    if len(image.shape) == 2:
        height, width = image.shape
        r_image = np.reshape(image, (height, width, 1))
    else: 
        r_image = image
    
    # 画像サイズの取得
    height, width, deepth = r_image.shape
    # 結果画像 
    f_img = np.zeros((height, width, deepth))

    # 正規化
    for z in range(0,deepth):
        z_image = r_image[:,:,z]
        # zチャンネルの一次元配列化
        img = np.array(z_image).flatten()
        # 平均と分散の算出
        mean = img.mean()
        std = np.std(img)

        # zチャンネルでの正規化
        for y in range(0,height):
            for x in range(0,width):
                if std < pow(10,-10):
                    f_img[y][x][z] = 127
                else:
                    value = round((z_image[y][x] - mean + s*std) / (2*s*std) * 255)
                    if value > 255:
                        f_img[y][x][z] = 255
                    elif value < 0:
                        f_img[y][x][z] = 0
                    else:
                        f_img[y][x][z] = value
        
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img
    
def list_median(value_list):
    """
    数値のリストから中央値を取得
    """
    med_i = round(len(value_list)/2)
    sort_list = sorted(value_list)
    if len(sort_list) % 2 == 1:
        return int(sort_list[med_i])
    else:
        return (int(sort_list[med_i-1]) + int(sort_list[med_i])) / 2

def median(image, kernel_shape=ImgStruct_Name.CIRCLE, ksize=4, nml_on=False):
    """
    画像にメディアンフィルタを適用する．
    ## <Args>
        - image: 入力画像
        - kernel_shape: カーネルの構造． ImgStruct_Name.CIRCLE:円形，ImgStruct_Name:矩形
        - ksize: カーネルサイズ(円の半径) 
        - nml_on: 正規化をかける(デフォルト: NML2をかけない！！！)

    ## <Returns>
        - f_img: メディアンフィルタをかけてからトップハット変換した画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    f_img = np.zeros((height, width))
    for y in range(0,height):
        for x in range(0,width):
            kernel_set = ImgStruct_Set(x,y,image.shape).get(kernel_shape, ksize)
            value_list = []
            for coor in kernel_set:
                coor_x, coor_y = coor
                value_list.append(image[coor_y][coor_x])
            med = list_median(value_list)
                
            f_img[y][x] = round(med)
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def tophat(image, kernel_shape=cv2.MORPH_ELLIPSE, ksize=(4,4), nml_on=True):
    """
    画像をトップハット変換します．
    ## <Args>
        - image: 入力画像  
        - shape: カーネルの構造．cv2のcv2.MORPH_ELLIPSE:楕円形(デフォルト)，cv2.MORPH_RECT:矩形   
        - ksize: カーネルサイズ．タプル，カーネルの構造に合わせて入力(デフォルト: 楕円形，(4,4))
        - nml_on: 正規化をかける(デフォルト: NML2をかける)

    ## <Returns>
        - f_img: トップハット変換した画像    
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    kernel = cv2.getStructuringElement(kernel_shape, ksize)
    f_img = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def median_tophat(image, median_ksize, tophat_ksize, nml_on=True):
    """
    画像にメディアンフィルタをかけてからトップハット変換を行う．
    このとき用いるカーネルの構造はメディアンフィルタもトップハット変換も円形
    ## <Args>
        - image: 入力画像
        - media_ksize: カーネルサイズ(円の半径)
        - tophat_ksize: カーネルサイズ(円の半径)
        - nml_on: 正規化をかける(デフォルト: NML2をかける)

    ## <Returns>
        - f_img: メディアンフィルタをかけてからトップハット変換した画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    f_m_img = median(image, ksize=median_ksize)
    f_img = tophat(f_m_img, ksize=(tophat_ksize,tophat_ksize), nml_on=False)
    if nml_on:
        f_img = normalization(f_img, 2)
    
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def sobel(image, kernel, nml_on):
    """
    ソーベルフィルタを適用する．
    負の勾配も考慮するため，出力されるデータの型はcv2.CV_64Fである．
    ## <Args>
        - image: 入力画像
        - kernel: カーネル．FilterKernel内のカーネルを選択(SBL○××)．
        - nml_on: 正規化をかける(デフォルト: NML2をかける)

    ## <Returns>
        - f_img: ソーベルフィルタを適用した画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    f_img = cv2.filter2D(image, -1, kernel)
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def sobel_mag(image, nml_on=True):
    """
    ソーベルフィルタマグニチュードを適用する．(M = sqrt(dx^2+dy^2))
    カーネルはx方向，y方向どちらも3×3のフィルタを使用する
    ## <Args>
        - image: 入力画像  
        - nml_on: 正規化をかける(デフォルト: NML2をかける)

    ## <Returns>
        - f_img: ソーベルフィルタを適用した画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    dx_img = sobel(image, FilterKernel.SBLX33, nml_on=False)
    dx_img = np.asarray(dx_img, dtype=np.float64).reshape((height, width, 1))
    dy_img = sobel(image, FilterKernel.SBLY33, nml_on=False)
    dy_img = np.asarray(dy_img, dtype=np.float64).reshape((height, width, 1))
    f_img = cv2.magnitude(dx_img, dy_img)
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def sobel_dir(image, nml_on=True):
    """
    ソーベルディレクションを適用する．(M = {255(arctan(dy/dx)+π)}/{2π})
    カーネルはx方向，y方向どちらも3×3のフィルタを使用する
    ## <Args>
        - image: 入力画像  
        - nml_on: 正規化をかける(デフォルト: NML2をかける)

    ## <returns>
        - f_img: ソーベルフィルタを適用した画像
    """
    dx_img = sobel(image, FilterKernel.SBLX33, nml_on=False)
    dy_img = sobel(image, FilterKernel.SBLY33, nml_on=False)
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    f_img = np.zeros((height, width))
    for y in range(0,height):
        for x in range(0,width):
            dx = dx_img[y][x]
            dy = dy_img[y][x]
            value = (math.atan2(dy, dx) + math.pi) * 255 / (2 * math.pi)
            f_img[y][x] = round(value)
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def laplacian(image, kernel, nml_on=True):
    """
    ラプラシアンフィルタを適用する．
    出力されるデータの型はcv2.CV_64Fである．
    ## <Args>
        - image: 入力画像
        - kernel: カーネル．FilterKernel内のLPL4(4近傍)またはLPL8(8近傍)
        - nml_on: 正規化をかける(デフォルト: NML2をかける)

    ## <Returns>
        - f_img: ラプラシアンフィルタを適用した画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    f_img = cv2.filter2D(image, -1, kernel)
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def mean(image, ksize=(3,3), nml_on=True):
    """
    平均化フィルタを適用する．
    ## <Args>
        - image: 入力画像
        - ksize: カーネルサイズ.タプル，デフォルト(3,3)
        - nml_on: 正規化をかける(デフォルト: NML2をかける)

    ## <Returns>
        - f_img: 平均化フィルタを適用した画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    f_img = cv2.blur(image, ksize)
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def gaussian(image, ksize, nml_on=True):
    """
    ガウシアンフィルタを適用する．
    ## <Args>
        - image: 入力画像
        - ksize: カーネルサイズ.3or5
        - nml_on: 正規化をかける(デフォルト: NML2をかける)

    ## <Returns>
        - f_img: ガウシアンフィルタを適用した画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    kernel = np.zeros((ksize,ksize))
    sigma = 0.3 * (int(ksize/2) - 1) + 0.8
    for y in range(int(-ksize/2), int(ksize/2)+1):
        for x in range(int(-ksize/2), int(ksize/2)+1):
            kernel[y + int(ksize/2)][x + int(ksize/2)] = \
                math.exp((-x*x-y*y)/(2*sigma*sigma)) / (2 * math.pi*sigma*sigma)
    f_img = cv2.filter2D(image, -1, kernel)
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def local_binary_pattern(image, p=1, nml_on=True):
    """
    ローカルバイナリパターンを適用する．
    ## <Args>
        - image: 入力画像
        - p: フィルタの種類．1: 3×3， 2: 5×5(一個飛ばし)
        - nml_on: 正規化をかける(デフォルト: NML2をかける)

    ## <Returns>
        -f_img: ローカルバイナリパターンを適用した画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    f_img = np.zeros((height, width))
    for y in range(1, height-p):
        for x in range(1, width-p):
            center_value = image[y][x]
            bp = np.zeros((2*p+1,2*p+1))
            for dy in range(-p, p+1, p):
                for dx in range(-p, p+1, p):
                    value = image[y+dy][x+dx]
                    if value > center_value:
                        bp[dx+p][dy+p] = 1
                    else:
                        bp[dx+p][dy+p] = 0
            value = bp[0][0] * 128 + bp[1*p][0] * 64 + bp[2*p][0] * 32\
                    + bp[2*p][1*p] * 16 + bp[2*p][2*p] * 8\
                    + bp[1*p][2*p] * 4 + bp[0][2*p] * 2 + bp[0][1*p]
            f_img[y][x] = value
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def median_lbp(image, median_ksize=1, lbp_p=1, nml_on=True):
    """
    画像にメディアンフィルタをかけてからローカルバイナリパターンを行う．
    このとき用いるカーネルの構造はメディアンフィルタは円形
    ## <Args>
        - image: 入力画像
        - media_ksize: カーネルサイズ(円の半径)
        - lbp_p: ローカルバイナリパターンのフィルタの種類
        - nml_on: 正規化をかける(デフォルト: NML2をかける)

    ## <Returns>
        - f_img: メディアンフィルタをかけてからローカルバイナリパターン
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    f_m_img = median(image, ksize=median_ksize)
    f_img = local_binary_pattern(f_m_img, lbp_p, nml_on=False)
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def exponential_tone_curve(image, gamma, nml_on=True):
    """
    指数対数型トーンカーブ
    ## <Args>
        - image: 入力画像
        - gamma: 255*(x/255)^(1/gammma)のパラメータ(現在2.0or0.5)
        - nml_on: 正規化をかける(デフォルト: NML2をかける)

    ## <Returns>
        - f_img: 指数対数型トーンカーブをかけた画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))

    f_img = np.zeros((height, width))
    for y in range(0, height):
        for x in range(0, width):
            value = 255*math.pow(image[y][x]/255, 1/gamma)
            f_img[y][x] = round(value)
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def sigmoid_tone_curve(image, b, c, nml_on=True):
    """
    S字型トーンカーブ
    ## <Args>
        - image: 入力画像
        - b, c: 255*sin((x/255 - b)π) + 1) / c のパラメータ(現在(b,c)=(0.0,2.0)or(1.0,2.0))
        - nml_on: 正規化をかける(デフォルト: NML2をかける)

    ## <Returns>
        - f_img: S字型トーンカーブをかけた画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    f_img = np.zeros((height, width))
    for y in range(0, height):
        for x in range(0, width):
            value = 255*(math.sin(math.pi*(image[y][x]/255 - b)) + 1) / c
            f_img[y][x] = round(value)
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def hist_flat(image, nml_on=True):
    """
    ヒストグラム平坦化
    ## <Args>
        - image: 入力画像
        - nml_on: 正規化をかける(デフォルト: NML2をかける)

    ## <Returns>
        - f_img: ヒストグラム平坦化をかけた画像
    """
    # 出力画像の用意
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))

    f_img = np.zeros((height, width))

    # 画素値LUT
    value_counter = [0] * 256

    # 画素値LUT穴埋め
    for y in range(0, height):
        for x in range(0, width):
            value = image[y][x]
            value_counter[value] = value_counter[value]+1
    
    # 平坦化後の各レベルの画素数を求める
    base_rate = math.ceil(height*width/256)

    # 出力濃度レベルのLUT
    lut = [0] * 256

    # 最小画素数のカウンタ
    min = 0

    # 出力濃度レベル
    level = 0

    # 画素数の累計カウンタ
    pixel_counter = 0

    # 出力濃度レベルのLUT穴埋め    
    for i in range(0, 256):
        pixel_counter = value_counter[i] + pixel_counter
        if pixel_counter >= base_rate:
            for j in range(min, i+1):
                lut[j] = level
            level = int(pixel_counter/base_rate) + level
            min = i + 1
            pixel_counter = pixel_counter % base_rate
        elif i == 255:
            for j in range(min, i+1):
                lut[j] = level
    
    # 平坦化処理
    for y in range(0, height):
        for x in range(0, width):
            f_img[y][x] = lut[image[y][x]]

    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def negaposi(image, nml_on=True):
    """
    ネガポジ反転
    ## <Args>
        - image: 入力画像
        - nml_on: 正規化をかける(デフォルト: NML2をかける)

    ## <Returns>
        - f_img: ネガポジ反転をかけた画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    f_img = cv2.bitwise_not(image)

    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def posterization(image, n, nml_on=True):
    """
    n段階のポスタリゼーション
    ## <Args>
        - image: 入力画像
        - n: パラメータ(現在10or20or30)
        - nml_on: 正規化をかける(デフォルト: NML2をかける)

    ## <Returns>
        - f_img: n段階のポスタリゼーションをかけた画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))

    f_img = np.zeros((height, width))
    for y in range(0, height):
        for x in range(0, width):
            n_value = math.floor(image[y][x] * n / 255)
            value = round(255 * n_value / (n-1))
            f_img[y][x] = value
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def solarization(image, nml_on=True):
    """
    ソラリゼーション
    ## <Args>
        - image: 入力画像
        - nml_on: 正規化をかける(デフォルト: NML2をかける)

    ## <Returns>
        - f_img: ソラリゼーションをかけた画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    f_img = np.zeros((height, width))
    for y in range(0, height):
        for x in range(0, width):
            const_value = 2 * math.pi / 255
            value = abs(round(math.sin(image[y][x]*const_value)*255))
            f_img[y][x] = value
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def emboss(image, kernel, nml_on=True):
    """
    エンボスフィルタを適用する．
    ## <Args>
        - image: 入力画像
        - kernel: カーネル．FilterKernel内のカーネルを選択(EMB××)．
        - nml_on: 正規化をかける(デフォルト: NML2をかける)

    ## <Returns>
        - f_img: ソーベルフィルタを適用した画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    f_img = cv2.filter2D(image, -1, kernel)
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def knn_ave(image, ksize, k, nml_on=True):
    """
    最近傍化フィルタ 
    ksize*ksizeの中から注目画素の画素値と近いk個の画素を選択し，その平均を出力する
    ## <Args>
        - image: 入力画像
        - ksize: 注目領域の半径(現在3or5)
        - k: 近傍k個の画素値(現在3or5)
        - nml_on: 正規化をかける(デフォルト: NML2をかける)

    ## <Returns>
        - f_img: 最近傍化フィルタをかけた画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    f_img = np.zeros((height, width))
    for y in range(ksize, height-ksize):
        for x in range(ksize, width-ksize):
            v = image[y][x]
            ii = 0
            value = 0
            value_list = []
            for dy in range(-ksize,ksize+1):
                for dx in range(-ksize,ksize+1):
                    value_list.append(image[y+dy][x+dx])
            sort_list = sorted(value_list)
            for i in range(0, len(value_list)):
                if sort_list[i] >= v:
                    ii = i
                    break
            if ii >= int(k/2) and ii < len(value_list) - int(k/2):
                for i in range(ii-int(k/2), ii+int(k/2)+1):
                    value = sort_list[i] + value 
            elif ii < len(value_list) - int(k/2):
                for i in range(0, k):
                    value = sort_list[i] + value
            else:
                for i in range(len(value_list)-k, len(value_list)):
                    value = sort_list[i] + value
            value = round(value / k)
            f_img[y][x] = value
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def bilateral(image, ksize, sigma1, sigma2, nml_on=True):
    """
    バイラテラルフィルタ．中心画素からの距離と画素値の差に応じて，
    ガウシアン分布に従う重みをつける平均化フィルタの一種
    ## <Args>
        - image: 入力画像
        - ksize: 注目領域の半径(現在3or5)
        - simga1, sigam2: パラメータ(現在(sigma1,sigma2)=(1.0,2.0))
        - nml_on: 正規化をかける(デフォルト: NML2をかける)

    ## <Returns>
        - f_img: n段階のポスタリゼーションをかけた画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    f_img = np.zeros((height, width))
    p = int(ksize)
    for y in range(p, height - p):
        for x in range(p, width - p):
            v = int(image[y][x])
            # 分子部分と分母部分それぞれ分けて計算する
            numerator = 0
            denominator = 0

            for dy in range(-p, p+1):
                for dx in range(-p, p+1):
                    vv = int(image[y+dy][x+dx])
                    coef1 = math.exp((-int(dx)*int(dx) -int(dy)*int(dy)) / (2*int(sigma1)*int(sigma1)))
                    coef2 = math.exp((-int((v-vv))*int((v-vv))) / (2*int(sigma2)*int(sigma2)))
                    coef = coef1 * coef2 
                    numerator = coef*vv + numerator
                    denominator = coef + denominator
            
            value = round(numerator/denominator)
            f_img[y][x] = value
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def black(image): 
    """
    黒い画像を返す
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    f_img = np.zeros((height, width))
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def marrHildreth(image, sigma=3, nml_on=True):
    """
    Marr-Hildreth Operator
    """

    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    size = int(2*(np.ceil(3*sigma))+1)

    x, y = np.meshgrid(np.arange(-size/2+1, size/2+1),
                       np.arange(-size/2+1, size/2+1))

    normal = 1 / (2.0 * np.pi * sigma**2)

    kernel = ((x**2 + y**2 - (2.0*sigma**2)) / sigma**4) * \
        np.exp(-(x**2+y**2) / (2.0*sigma**2)) / normal  # LoG filter

    kern_size = kernel.shape[0]
    log = np.zeros_like(image, dtype=float)

    # applying filter
    for i in range(image.shape[0]-(kern_size-1)):
        for j in range(image.shape[1]-(kern_size-1)):
            window = image[i:i+kern_size, j:j+kern_size] * kernel
            log[i, j] = np.sum(window)

    log = log.astype(np.int64, copy=False)

    f_img = np.zeros_like(log)

    # computing zero crossing
    for i in range(log.shape[0]-(kern_size-1)):
        for j in range(log.shape[1]-(kern_size-1)):
            if log[i][j] == 0:
                if (log[i][j-1] < 0 and log[i][j+1] > 0) or (log[i][j-1] < 0 and log[i][j+1] < 0) or (log[i-1][j] < 0 and log[i+1][j] > 0) or (log[i-1][j] > 0 and log[i+1][j] < 0):
                    f_img[i][j] = 255
            if log[i][j] < 0:
                if (log[i][j-1] > 0) or (log[i][j+1] > 0) or (log[i-1][j] > 0) or (log[i+1][j] > 0):
                    f_img[i][j] = 255
    if nml_on:
        f_img = normalization(f_img, 2)
    return f_img

def gammaCorrection(image, gamma=1.0, nml_on=True):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    f_img = np.empty(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            f_img[i,j] = cv2.LUT(np.array(image[i,j], dtype = np.uint8), table)
    if nml_on:
        f_img = normalization(f_img, 2)
    return f_img

def enhancePassFilter(image, d, k1=0.5, k2=0.75):
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    
    f = np.fft.fft2(image)
    dftShift = np.fft.fftshift(f)
    hDftShift, wDftShift = dftShift.shape
    halfHeightDftShift = np.fix(hDftShift/2.0)
    halfWidthDftShift = np.fix(wDftShift/2.0)
    tmpImg1 = np.zeros_like(dftShift, dtype=np.complex64)
    tmpImg2 = np.zeros_like(dftShift, dtype=np.complex64)
    for i in range(0,hDftShift):
        for j in range(0, wDftShift):
            distance = np.square(i-halfHeightDftShift)+np.square(j-halfWidthDftShift)
            tmpImg1[i][j] = 1-np.exp((-1)*distance/(2*np.square(d)))
            tmpImg2[i][j] = k1 + k2 * tmpImg1[i][j]
    tmpImg2 = dftShift * tmpImg2
    backIfft2 = np.abs(np.fft.ifft2(np.fft.ifftshift(tmpImg2), axes=(0,1)))
    backIfft2 = np.asarray(backIfft2, dtype=np.uint8).reshape(backIfft2.shape)
    f_img = cv2.equalizeHist(backIfft2)
    return f_img

def adaptiveGammaCorrection(image, t = 3, nml_on=True):
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    img = image/255.0
    mean = np.mean(img)
    variance = np.var(img)
    gamma = np.exp((1-(mean+variance))/2.0)
    if 4*variance <= 1/t:
        gamma = -np.log2(variance)
    
    heavisideCoefficient = 1
    if mean >= 0.5:
        heavisideCoefficient = 0
    expVal = np.power(img, gamma)
    k = expVal + (1 - expVal) * np.power(mean, gamma)
    c = 1.0/(1+heavisideCoefficient*(k-1))
    f_img = c*np.power(img, gamma)
    if nml_on:
        f_img = normalization(f_img, 2)
    return f_img
    
'''
input_path = "../dataset/without_black_area_crop/image/krhaa001_n_N1_from3072.png"
image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
# nml_image = bilateral(image, 3, 1, 2)
nml_image = marrHildreth(image)
output_path = "test_mar.png"
cv2.imwrite(output_path, nml_image)
'''