# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
# Name          : perceptron
# Version       : 0.1.0
# Author        : yxf
# Language      : Python 3.6
# Start time    : 2017-07-03 14:50
# End time      :
# Function      : 
# Operation     :
#------------------------------------------------------------------------------

# System Moduls
import os
import sys
import math

# External Moduls
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Custom Moduls
from cv_tools import getHogFeature
from load_mnist import DataUtils


class Perceptron() :
    def __init__(self) :
        self._w = self._b = None

    # lr 梯度下降法中的学习速率
    # epoch 梯度下降法中的迭代上限
    def fit(self , x , y , lr=0.001 , epoch=1000) :
        x = np.array(x , np.float32)
        y = np.array(y , np.float32)
        self._w = np.zeros(x.shape[1])
        self._b = 0.0

        for _ in range(epoch) :
            # compute w*x+b
            err = -y * self.predict(x , True)
            # 选出使得损失函数最大的样本
            idx = np.argmax(err)
            # 若该样本被正确分类，则结束训练
            if err[idx] < 0 :
                break
            # 否则，让参数沿着负梯度方向走一步
            delta = lr * y[idx]
            self._w += delta * x[idx]
            self._b += delta
        
    def trainMNIST(self , train_x , train_y , lr=0.001 , epoch=1000) :
        train_x = np.array(train_x , np.float32)
        train_y = np.array(train_y , np.float32)
        size = train_x.shape[0]
        self._w = np.zeros(train_x.shape[1])
        self._b = 0.0

        for i in range(size) :
            x = train_x[i]
            y = train_y[i]
            for _ in range(epoch) :
                err = - y * self.predict(x , True)
                if err > 0 :
                    break
                delta = lr * y
                self._w += delta * x    # w <-- w + lr * y * x
                self._b += delta        # b <-- b + lr * y
        return self._w , self._b

    def predict(self , x , raw=False) :
        x = np.asarray(x , np.float32)
        y_pred = x.dot(self._w) + self._b
        if raw :
            return y_pred
        rst = np.sign(y_pred).astype(np.float32)
        return rst


if "__main__" == __name__ :
    pre_path = "../dataset/MNIST/"
    out_path = "./unzipdata/"
    train_file_imgs = "train-images-idx3-ubyte"
    train_file_lbls = "train-labels-idx1-ubyte"
    test_file_imgs = "t10k-images-idx3-ubyte"
    test_file_lbls = "t10k-labels-idx1-ubyte"
    """
    if not os.path.isdir(out_path) :
        os.mkdir(out_path)
    os.system("tar -zxvf " + pre_path + train_file_imgs + ".tar.gz -C " + out_path)
    os.system("tar -zxvf " + pre_path + train_file_lbls + ".tar.gz -C " + out_path)
    os.system("tar -zxvf " + pre_path + test_file_imgs + ".tar.gz -C " + out_path)
    os.system("tar -zxvf " + pre_path + test_file_lbls + ".tar.gz -C " + out_path)
    """

    train_imgs = DataUtils(filename=out_path + train_file_imgs).getImage()
    train_lbls = DataUtils(filename=out_path + train_file_lbls).getLabel()
    test_imgs = DataUtils(filename=out_path + test_file_imgs).getImage()
    test_lbls = DataUtils(filename=out_path + test_file_lbls).getLabel()
    """
    print(train_imgs.shape)
    print(np.reshape(train_imgs[0] , (28 , 28)))
    print(train_lbls.shape)
    print(train_lbls[0])
    print(test_imgs.shape)
    print(test_lbls.shape)
    """
    train_x = list()
    train_y = list()
    test_x = list()
    test_y = list()
    for i in range(train_lbls.shape[0]) :
        if 3 == int(train_lbls[i]) or 4 == int(train_lbls[i]) :
            tmp = getHogFeature(np.array(train_imgs[i]))
            train_x.append(tmp)
            train_y.append(train_lbls[i])
    for i in range(test_lbls.shape[0]) :
        if 3 == int(test_lbls[i]) or 4 == int(test_lbls[i]) :
            tmp = getHogFeature(np.array(test_imgs[i]))
            test_x.append(tmp)
            test_y.append(test_lbls[i])
    train_x = np.reshape(np.array(train_x) , (-1 , 324))
    train_y = np.array(train_y)
    test_x = np.reshape(np.array(test_x) , (-1 , 324))
    test_y = np.array(test_y)
    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)
    pp = Perceptron()
    w , b = pp.trainMNIST(train_x , train_y)
    preds = list()
    for i in range(test_x.shape[0]) :
        prd = pp.predict(test_x[i] , True)
        preds.append((prd , test_y[i]))
    print(preds)
