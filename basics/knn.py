# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
# Name          : knn
# Version       : 0.1.0
# Author        : yxf
# Language      : Python 3.6
# Start time    : 2017-06-23 14:50
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
from load_mnist import DataUtils as DU


class KNNClassifier() :
    def __init__(self , k=3) :
        self._k = k

    def _euclideanDistance(self , sample , dataset) :
        num = dataset.shape[0]
        diff_mat = np.tile(sample , (num,1)) - dataset
        sq_diff_mat = diff_mat ** 2
        sum_diff_mat = sq_diff_mat.sum(axis=1)
        distances = sum_diff_mat ** 0.5
        return distances

    def _calcDistance(self , sample , dataset , dist_type=None) :
        result = None
        if None==dist_type or "ecd"==dist_type.lower() :
            result = self._euclideanDistance(sample , dataset)
        return result

    def _classify(self , sample , train_x , train_y) :
        distances = self._calcDistance(sample , train_x)
        sorted_dists = distances.argsort()
        class_count = dict()
        for i in range(self._k) :
            label = train_y[sorted_dists[i]]
            class_count[label] = class_count.get(label , 0) + 1
        sorted_class_list = sorted(class_count.items() , key=lambda val:val[1] , reverse=True)
        res_class = sorted_class_list[0][0]
        return res_class

    def classify(self , train_x , train_y , test_x , test_y=None) :
        result = list()
        if not (isinstance(train_x , np.ndarray) and isinstance(train_y , np.ndarray) and isinstance(test_x , np.ndarray)) :
            try :
                train_x = np.array(train_x)
                train_y = np.array(train_y)
                test_x = np.array(sample)
            except Exception as e :
                print(e)
                return False 

        p_count = 0.0
        for i in range(test_x.shape[0]) :
            value = self._classify(test_x[i] , train_x , train_y)
            p_count += 1 if value==test_y[i] else 0
            p = p_count/(i+1)*100
            print(str(p) + "%\t\t" + str(i+1) + "/" + str(test_x.shape[0]))
            result.append(value)


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

    train_imgs = DU(filename=out_path + train_file_imgs).getImage()
    train_lbls = DU(filename=out_path + train_file_lbls).getLabel()
    test_imgs = DU(filename=out_path + test_file_imgs).getImage()
    test_lbls = DU(filename=out_path + test_file_lbls).getLabel()
    print(train_imgs.shape)
    print(train_lbls.shape)
    print(test_imgs.shape)
    print(test_lbls.shape)
    print(len(np.shape(test_imgs)))
    kcf = KNNClassifier(k=3)
    kcf.classify(train_imgs , train_lbls , test_imgs , test_lbls)
