import os
import cv2


def save_images(savepath, filenamelist, imagelist):
    """
    指定ディレクトリに画像リストの画像を保存する
    ### <Args>
        - savepath: 保存するディレクトリのパス
        - filenamelist: ファイル名リスト
        - imagelist: 画像リスト

    ### <Returns>
        なし
    """
    for i, fn in enumerate(filenamelist):
        filename = os.path.join(savepath, fn)
        testimage = imagelist[i]
        cv2.imwrite(filename, testimage)

def make_folder(root_dir, folder_name, cmd_mode=False):
    """
    フォルダを作成します．上書き保存をするか選択できる．
    上書き保存しない場合，folder_name(数字)の新しいファイルが作成される．
    デフォルトではコマンドラインで聞かれない．cmd_modeがTrueのとき対話形式．
    ### <Args>
        - root_dir: 作成したフォルダを作成する場所
        - folder_name: フォルダの名前
        - cmd_mode: フォルダの書き出しについてコマンドライン上で聞くか
        
    ### <Returns>
        - resDir: 作成されたフォルダのパス
    """
    resDir = root_dir + folder_name + '/'
    if os.path.exists(resDir) and cmd_mode:
        print("結果フォルダ{0}が既に存在しています．".format(resDir))
        print("上書き保存しますか？[y/n]")
        c = input()
        if c == 'y' or c == 'yes' or c == 'Yes':
            pass
        else:
            print("新しいファイルを作成します．")
            i = 0
            while os.path.exists(resDir):
                i = i + 1
                new_folder_name = folder_name + '({0})'.format(i)
                resDir = root_dir + new_folder_name + '/'
    # フォルダの作成            
    os.makedirs(resDir, exist_ok=True)
        
    return resDir
