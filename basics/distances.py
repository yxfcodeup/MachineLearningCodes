# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
# Name          : distances
# Version       : 0.1.0
# Author        : yxf
# Language      : Python 3.6
# Start time    : 2017-06-23 17:00
# End time      :
# Function      : 
# Operation     :
#------------------------------------------------------------------------------

# System Moduls
import os
import sys
import math
import logging

# External Moduls
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Custom Moduls


def euclideanDistance(arr_x , arr_y) :
    if arr_x.shape != arr_y.shape :
        logging.error("arr_x and arr_y must be the same shape!")
        return False
    diff_mat = arr_x - arr_y
    sq_diff_mat = diff_mat ** 2
    sum_diff_mat = sq_diff_mat.sum(axis=1)
    distances = sum_diff_mat ** 0.5
    return distances


if "__main__" == __name__ :
    a = np.array([[1,2,3] , [4,5,6] , [7,8,9]])
    b = np.array([[2,3,4] , [5,6,7] , [8,9,1]])
    r1 = euclideanDistance(a , b)
    print(r1)
