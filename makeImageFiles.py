# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 21:31:20 2018

@author: taishi-o
"""

import numpy as np
import cv2
import os

#----------------
# define target .mp4 files
#----------------
mov_names = ["asobi_asobase11.mp4",
            "gochiusa_01.mp4",
            "hataraku_saibou08.mp4",
            "planet_with08.mp4",
            "revu_star07.mp4"]

#----------------
# define functions
#----------------
def make_files(mov_names):
    paths = []
    for num in range(len(mov_names)):
        mov_name = mov_names[num]
        basepath = "./picture/" + (mov_name.replace(".mp4", ""))
        
        os.makedirs(basepath + "/color/")
        os.makedirs(basepath + "/sketch/")
        
        paths.append(basepath)
        
    return paths
        
def load_movies(mov_names):
    movs = []
    
    for num in range(len(mov_names)):
        mov_name = mov_names[num]
        movs.append(cv2.VideoCapture("./movie/" + mov_name))
        
    return movs

def save_images(movs, paths):
    for movnum in range(len(movs)):
        mov = movs[movnum]
        savepath = paths[movnum]
        print("Now %s\n" %savepath)
        
        cnt = 0
        frame_base = 0
        while(mov.isOpened()):
            ret, frame = mov.read()
            if ret:
                if cnt==0: #the first frame
                    frame_base = frame
                    cv2.imwrite(savepath + "/color/%06d.png" %cnt, 
                                cv2.resize(frame_base, (320, 176)))
                    save_sketch(frame_base, savepath + "/sketch/%06d.png" %cnt)
                    cnt += 1
                else:
                    diff = frame_base.astype(np.float32) - frame.astype(np.float32) #difference between the base frame
                    if abs(np.mean(diff)) > .2:
                        frame_base = frame #renew the base frame
                        cv2.imwrite(savepath + "/color/%06d.png" %cnt, 
                                    cv2.resize(frame_base, (320, 176))) #save
                        save_sketch(frame_base, savepath + "/sketch/%06d.png" %cnt)
                        cnt += 1
                    else:
                        pass
            else:
                break
        
        mov.release()
        
def save_sketch(img, path):
    kernel = np.ones((3,3), np.uint8)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dil = cv2.dilate(gray, kernel, iterations=1)
    diff = dil - gray
    
    cv2.imwrite(path, cv2.resize(255 - diff, (320, 176)))
    
        

#----------------
# main
#----------------
def main(mov_names):
    paths = make_files(mov_names)
    movs = load_movies(mov_names)
    save_images(movs, paths)
    
    return 0

if __name__ == "__main__":
    main(mov_names)

