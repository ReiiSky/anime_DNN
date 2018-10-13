# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 11:00:31 2018

@author: taishi-o
"""

import os
import numpy as np

import myNetwork
import myDatasetClass
import myLossfun

import chainer
from chainer.datasets import split_dataset_random
from chainer import iterators
from chainer import optimizers
from chainer.dataset import concat_examples
from chainer.cuda import to_cpu
import chainer.functions as F
from chainer import serializers


#--------------------
# Define parameters
#--------------------
file_list = ["./picture/gochiusa_01/",
             "./picture/hataraku_saibou08/",
             "./picture/planet_with08/",
             "./picture/revu_star07/"]
savefile = "./result/20181010_UNET_02/"
batchsize = 6
gpu_id = 0
max_epoch = 500
reduce = 3

#--------------------
# Set up dataset
#--------------------
dataset = myDatasetClass.myImageDataset(file_list, reduce=reduce)

train, valid = split_dataset_random(
        dataset, int(len(dataset)*0.8), seed=0)

train_iter = iterators.SerialIterator(train, batchsize)
valid_iter = iterators.SerialIterator(
        valid, batchsize, repeat=False, shuffle=False)

#--------------------
# Set up network
#--------------------
net = myNetwork.UNET_02()

if gpu_id >= 0:
    net.to_gpu(gpu_id)

optimizer = optimizers.Adam().setup(net)

#--------------------
# Learning iteration
#--------------------
os.makedirs(savefile+"/models/")

while train_iter.epoch < max_epoch:
    
    train_batch = train_iter.next()
    x, t = concat_examples(train_batch, gpu_id)
    
    y = net(x)
    
    #loss = F.mean_squared_error(y, t)
    loss = myLossfun.l1l2_norm_error(y, t, lam=3)
    
    net.cleargrads()
    loss.backward()
    
    optimizer.update()
    
    # Check
    if train_iter.is_new_epoch:
        print('epoch:{:02d}   train_loss:{:.04f} '.format(
            train_iter.epoch, float(to_cpu(loss.data))), end='')
        
        valid_losses = []
        while True:
            valid_batch = valid_iter.next()
            x_valid, t_valid = concat_examples(valid_batch, gpu_id)
            
            with chainer.using_config('train', False), \
                    chainer.using_config('enable_backprop', False):
                y_valid = net(x_valid)

            loss_valid = F.mean_squared_error(y_valid, t_valid)
            valid_losses.append(to_cpu(loss_valid.array))
            
            if valid_iter.is_new_epoch:
                valid_iter.reset()
                break
            
        print('    val_loss:{:.04f}'.format(
            np.mean(valid_losses)))
        
        # Save model
        if train_iter.epoch % 10 == 0:
            serializers.save_npz(savefile+"/models/snapshot_{:d}".format(
                    train_iter.epoch), net)

