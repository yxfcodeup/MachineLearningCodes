# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
# Name          : load_mnist
# Version       : 0.1.0
# Author        : yxf
# Language      : Python 3.6
# Start time    : 2017-06-21 16:00
# End time      :
# Function      : 
# Operation     :
#------------------------------------------------------------------------------

# System Moduls
import os
import sys
import struct

# External Moduls
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Custom Moduls


class DataUtils(object) :
    def __init__(self , filename=None , outpath=None) :
        self._filename = filename
        self._outpath = outpath
        self._tag = ">"
        self._twoBytes = "II"
        self._fourBytes = "IIII"
        self._pictureBytes = "784B"
        self._labelByte = "1B"
        self._twoBytes2 = self._tag + self._twoBytes
        self._fourBytes2 = self._tag + self._fourBytes
        self._pictureBytes2 = self._tag + self._pictureBytes
        self._labelByte2 = self._tag + self._labelByte

    def getImage(self) :
        bin_file = open(self._filename , "rb")
        buf = bin_file.read()
        bin_file.close()
        idx = 0
        num_magic , num_imgs , num_rows , num_cols = struct.unpack_from(self._fourBytes2 , buf , idx)
        idx += struct.calcsize(self._fourBytes)
        images = list()
        for i in range(num_imgs) :
            img_val = struct.unpack_from(self._pictureBytes2 , buf , idx)
            idx += struct.calcsize(self._pictureBytes2)
            img_val = list(img_val)
            for j in range(len(img_val)) :
                img_val[j] = 1 if img_val[j] > 1 else 0
            images.append(img_val)
        return np.array(images)

    def getLabel(self) :
        bin_file = open(self._filename , "rb")
        buf = bin_file.read()
        bin_file.close()
        idx = 0
        magic , num_items = struct.unpack_from(self._twoBytes2 , buf , idx)
        idx += struct.calcsize(self._twoBytes2)
        labels = list()
        for i in range(num_items) :
            im = struct.unpack_from(self._labelByte2 , buf , idx)
            idx += struct.calcsize(self._labelByte2)
            labels.append(im[0])
        return np.array(labels)

    def outImage(self , images , labels) :
        m , n = np.shape(images)
        for i in range(1) :
            img = np.array(images[i])
            img = img.reshape(28 , 28)
            out_img = str(i) + "_" + str(labels[i]) + ".jpg"
            plt.imsave(out_img , img)


if "__main__" == __name__ :
    pre_path = "./"
    out_path = "./unzipdata/"
    train_file_imgs = "train-images-idx3-ubyte"
    train_file_lbls = "train-labels-idx1-ubyte"
    test_file_imgs = "t10k-images-idx3-ubyte"
    test_file_lbls = "t10k-labels-idx1-ubyte"
    if not os.path.isdir(out_path) :
        os.mkdir(out_path)
    os.system("tar -zxvf " + pre_path + train_file_imgs + ".tar.gz -C " + out_path)
    os.system("tar -zxvf " + pre_path + train_file_lbls + ".tar.gz -C " + out_path)
    os.system("tar -zxvf " + pre_path + test_file_imgs + ".tar.gz -C " + out_path)
    os.system("tar -zxvf " + pre_path + test_file_lbls + ".tar.gz -C " + out_path)

    #train_imgs = DataUtils(filename=out_path + train_file_imgs).getImage()
    #train_lbls = DataUtils(filename=out_path + train_file_lbls).getLabel()
    #test_imgs = DataUtils(filename=out_path + test_file_imgs).getImage()
    #test_lbls = DataUtils(filename=out_path + test_file_lbls).getLabel()
    #DataUtils().outImage(train_imgs , train_lbls)
    #DataUtils().outImage(test_imgs , train_lbls)
