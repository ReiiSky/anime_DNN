# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 10:50:31 2018

@author: taishi-o
"""

import cv2
import numpy as np

import myNetwork

from chainer import serializers


#----------------
# Define dataset
#----------------
file = "./picture/asobi_asobase11/"
num = np.arange(677,695,1)

test = []
for imgnum in range(num.shape[0]-2):
    lr_00 = cv2.imread(file+"/color/{:06d}.png".format(
            num[imgnum]), -1).astype(np.float32).transpose(2,0,1)
    lr_01 = np.array([cv2.imread(file+"/sketch/{:06d}.png".format(
            num[imgnum]+1), -1).astype(np.float32).transpose(0,1)])
    lr_02 = cv2.imread(file+"/color/{:06d}.png".format(
            num[imgnum]+2), -1).astype(np.float32).transpose(2,0,1)
    
    lr = np.array([lr_00[2,:,:], lr_00[1,:,:], lr_00[0,:,:],
                   lr_01[0,:,:],
                   lr_02[2,:,:], lr_02[1,:,:], lr_02[0,:,:]])
    
    test.append(lr)

test = np.array(test)

#----------------
# Inference
#----------------
model = myNetwork.UNET()

serializers.load_npz("./result/20180925_lossmod/models/snapshot_270", model)

y = model(test)
y = y.array


#----------------
# Save
#----------------
for imgnum in range(num.shape[0]-2):
    img = y[imgnum,:,:,:].transpose(1,2,0)
    img = np.clip(img, 0, 255)
    cv2.imwrite("inference_{:03d}.png".format(imgnum), img[:,:,::-1].astype(np.uint8))

