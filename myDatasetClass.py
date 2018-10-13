# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 22:44:31 2018

@author: taishi-o
"""

import numpy as np
import cv2

import chainer

class myImageDataset(chainer.dataset.DatasetMixin):
    
    def __init__(self, dir_list, reduce):
        
        pairs = []
        cnt = 0
        
        for dirnum in range(len(dir_list)):
            dirname = dir_list[dirnum]
            with open(dirname + "/scene.txt") as f:
                l = f.readlines()
                for imgnum in range(len(l)):
                    seq = l[imgnum].split("-")
                    st, en = int(seq[0]), int(seq[1])
                    seq_len = en - st + 1
                    for seqnum in range(seq_len-2):
                        if cnt % reduce == 0:
                            pairs.append([dirname + "/color/%06d.png" %(st+seqnum),
                                          dirname + "/sketch/%06d.png" %(st+seqnum+1),
                                          dirname + "/color/%06d.png" %(st+seqnum+2),
                                          dirname + "/color/%06d.png" %(st+seqnum+1)])
                        else:
                            pass
                        
                        cnt += 1
                    
        self.pairs = pairs
        
    def __len__(self):
        return len(self.pairs)
    
    def get_image(self, filenames):
        print(filenames[0])
        lr_00 = cv2.imread(filenames[0], -1).astype(np.float32).transpose(2,0,1)
        lr_01 = np.array([cv2.imread(filenames[1], -1).astype(np.float32).transpose(0,1)])
        lr_02 = cv2.imread(filenames[2], -1).astype(np.float32).transpose(2,0,1)
        
        lb_00 = cv2.imread(filenames[3], -1).astype(np.float32).transpose(2,0,1)
        
        lr = np.array([lr_00[2,:,:], lr_00[1,:,:], lr_00[0,:,:],
                       lr_01[0,:,:],
                       lr_02[2,:,:], lr_02[1,:,:], lr_02[0,:,:]])
        
        lb = np.array([lb_00[2,:,:], lb_00[1,:,:], lb_00[0,:,:]])
        
        return lr, lb
        
    def get_example(self, i):
        filenames = self.pairs[i]
        images = self.get_image(filenames)
        
        return images

