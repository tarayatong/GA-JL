# coding: utf-8
#

import os

txtRoot = r'/home/liujing/Code/data/SAITD/SAITD/txt_train_test/126_129/val.txt'  # /home/liujing/Code/data/SAITD/SAITD/txt_train_test/110_131/train.txt
imgRoot = r'/home/liujing/Code/data/SAITD/SAITD/110/JPEGImages'
#/home/zangtao/Code/data/113/0.png 604,342,608,346,0
with open(txtRoot,"r") as F:
    F = F.readlines()
    txts = [txt.strip() for txt in F]
    for txt in txts:
        txtSplit = txt.split("/home/zangtao/Code/data/")[1].split("/")
        ImgId = txtSplit[0]
        ImgName =txtSplit[1].split(" ")[0]  #84.png
        label = txtSplit[1].split(".png")[1]  # 317,145,321,149,0 311,205,315,210,0

        #print("#"+ImgId+"  #"+ImgName +"  #"+label)
        wTxt = '/home/liujing/Code/data/SAITD/SAITD/'+ImgId+'/JPEGImages/'+ImgName +label
        print(wTxt)
        with open(r'/home/liujing/Code/data/SAITD/SAITD/txt_train_test/126_129/val_lj.txt', "a+") as f:
            f.write(wTxt+"\n")


