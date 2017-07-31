import os
import sys
#External Libs
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def getHogFeature(img) :
    hog = cv2.HOGDescriptor("./hog.xml")
    img = np.reshape(img , (28,28))
    cv_img = img.astype(np.uint8)
    hog_feature = hog.compute(cv_img)
    return hog_feature
