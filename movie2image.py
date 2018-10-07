# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 22:39:55 2018

@author: taishi-o
"""

import numpy as np
import cv2
import os

#----------------
#define parameter
#----------------
mov_name = "asobi_asobase11.mp4"

#----------------
#load movie file
#----------------
mov = cv2.VideoCapture("./movie/" + mov_name)

#----------------
#make save file for images
#----------------
basepath = "./picture/" + (mov_name.replace(".mp4", ""))
os.mkdir(basepath)
savepath = "./picture/" + (mov_name.replace(".mp4", "") + "/color/")
os.mkdir(savepath)

#----------------
#save each frame as .png
#----------------
cnt = 0
frame_base = 0
while(mov.isOpened()):
    ret, frame = mov.read()
    if ret:
        if cnt==0: #the first frame
            frame_base = frame
            cv2.imwrite(savepath + "/%06d.png" %cnt, 
                        cv2.resize(frame_base, (320, 180)))
            cnt += 1
        else:
            diff = frame_base.astype(np.float32) - frame.astype(np.float32) #difference between the base frame
            if abs(np.mean(diff)) > .2:
                frame_base = frame #renew the base frame
                cv2.imwrite(savepath + "/%06d.png" %cnt, 
                            cv2.resize(frame_base, (320, 180))) #save
                cnt += 1
            else:
                pass
    else:
        break

mov.release()
