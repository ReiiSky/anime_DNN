# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 21:30:44 2018

@author: taishi-o
"""

import cv2
import numpy as np
import glob
import os

#----------------
#define file path
#----------------
basename = "./picture/asobi_asobase11/"
file = basename + "/color/"
savefile = basename + "/sketch/"

name_list = glob.glob(file + "*.png")
os.mkdir(savefile)

#----------------
#extract line
#----------------
kernel = np.ones((3,3), np.uint8)

for imgnum in range(len(name_list)):
    #load target image
    img = cv2.imread(name_list[imgnum], cv2.IMREAD_GRAYSCALE).astype(np.float32)
    #dilation
    dil = cv2.dilate(img, kernel, iterations=1)
    #diff
    diff = dil - img
    
    cv2.imwrite(savefile + name_list[imgnum].split("\\")[-1],
                255 - diff)
