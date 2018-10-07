# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 23:00:33 2018

@author: taishi-o
"""

import sys
import numpy as np
import cv2

#------------------
# Load images
#------------------
file = "./picture/asobi_asobase11/"
sketchnum = np.arange(678,694,1)
infnum = np.arange(0, 16, 1)

#------------------
# Define functions
#------------------
def myFillImages(src):
    dst = np.array(src)
    
    # Mask
    h, w = dst.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask_for_fill = np.zeros((h+2, w+2), dtype=np.uint8)
    mask[src==0] = 1 # 1: Done, 0: Undone
    
    # Fill
    label = 1
    for i in range(h):
        for j in range(w):
            if mask[i, j] == 0: # If undone
                cv2.floodFill(
                        dst, mask_for_fill, (j,i), label)
                mask[mask_for_fill[1:1+h, 1:1+w]==1] = 1
                label += 1
                
                if np.sum(mask==0) == 0:
                    break
                
                if label >= 255:
                    print("label becomes larger than 255!!")
                    sys.exit()
                    
    return label, dst

def spreadColor(src, fill, label):
    dst = np.array(src)
    
    for labelnum in range(label):
        med_B = np.median(src[:,:,0][fill==labelnum])
        med_G = np.median(src[:,:,1][fill==labelnum])
        med_R = np.median(src[:,:,2][fill==labelnum])
        
        dst[:,:,0][fill==labelnum] = med_B
        dst[:,:,1][fill==labelnum] = med_G
        dst[:,:,2][fill==labelnum] = med_R
        
    return dst

def combineSketchInfSpread(spread, inference, sketch, thresh):
    dst = np.array(spread)
    
    dst[:,:,0][thresh==0] = inference[:,:,0][thresh==0]
    dst[:,:,1][thresh==0] = inference[:,:,1][thresh==0]
    dst[:,:,2][thresh==0] = inference[:,:,2][thresh==0]
    
    dst = dst.astype(np.float32)
    sketch = sketch.astype(np.float32)
    
    sketch_inv = 255 - sketch
    dst[:,:,0] -= sketch_inv
    dst[:,:,1] -= sketch_inv
    dst[:,:,2] -= sketch_inv
    
    dst = np.clip(dst, 0, 255).astype(np.uint8)
    
    return dst

#------------------
# Main
#------------------
for imgnum in range(sketchnum.shape[0]):
    sketch = cv2.imread(file + "sketch/{:06d}.png".format(
            sketchnum[imgnum]), -1)
    inference = cv2.imread("./inference_{:03d}.png".format(
            infnum[imgnum]), -1)
    # Threshold sketch img
    ret, thresh = cv2.threshold(sketch, 240, 255, cv2.THRESH_BINARY)
    # Fill
    label, fill = myFillImages(thresh)
    # Spread
    spread = spreadColor(inference, fill, label)
    # Combine with inference
    result = combineSketchInfSpread(spread, inference, sketch, thresh)
    
    # Save
    cv2.imwrite("test{:03d}.png".format(imgnum), result)
    