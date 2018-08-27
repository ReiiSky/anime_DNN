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
mov_name = "revu_star07.mp4"

#----------------
#load movie file
#----------------
mov = cv2.VideoCapture("./movie/" + mov_name)

#----------------
#make save file for images
#----------------
savepath = "./picture/" + (mov_name.replace(".mp4", ""))
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
            h, w = frame.shape[:2]
            frame_base = frame
            cv2.imwrite(savepath + "/%06d.png" %cnt, 
                        cv2.resize(frame_base, (int(w/2), int(h/2))))
            cnt += 1
        else:
            diff = np.abs(frame_base.astype(np.float32) - frame.astype(np.float32)) #difference between the base frame
            if np.mean(diff) > 0.2:
                frame_base = frame #renew the base frame
                cv2.imwrite(savepath + "/%06d.png" %cnt, 
                            cv2.resize(frame_base, (int(w/2), int(h/2)))) #save
                cnt += 1
            else:
                pass
    else:
        break

mov.release()
