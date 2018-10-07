# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 16:16:11 2018

@author: taishi-o
"""

import chainer
import chainer.links as L
import chainer.functions as F


#--------------
#network
#   https://qiita.com/taizan/items/cf77fd37ec3a0bef5d9d
#--------------
class Encode(chainer.Chain):
    def __init__(self, in_size, out_size):
        super(Encode, self).__init__()
        
        W = chainer.initializers.HeNormal()
        nobias = True
        
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                    in_size, out_size, ksize=3, stride=1, pad=1,
                    nobias=nobias, initialW=W)
            self.bn1 = L.BatchNormalization(out_size)
            
            self.conv2 = L.Convolution2D(
                    out_size, out_size, ksize=4, stride=2, pad=1,
                    nobias=nobias, initialW=W)
            self.bn2 = L.BatchNormalization(out_size)
        
    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        
        return h
    
class Decode(chainer.Chain):
    def __init__(self, in_size, out_size):
        super(Decode, self).__init__()
        
        W = chainer.initializers.HeNormal()
        nobias = True
        
        with self.init_scope():
            self.dconv1 = L.Deconvolution2D(
                    in_size, out_size, ksize=4, stride=2, pad=1,
                    nobias=nobias, initialW=W)
            self.bn1 = L.BatchNormalization(out_size)
            
            self.dconv2 = L.Convolution2D(
                    in_size, out_size, ksize=3, stride=1, pad=1,
                    nobias=nobias, initialW=W)
            self.bn2 = L.BatchNormalization(out_size)
            
    def __call__(self, x, y):
        h = F.relu(self.bn1(self.dconv1(x)))
        h = F.concat([h, y])
        h = F.relu(self.bn2(self.dconv2(h)))
        
        return h


class UNET(chainer.Chain):
    
    def __init__(self):
        super(UNET, self).__init__()
        
        W = chainer.initializers.HeNormal()
        nobias = True
        
        with self.init_scope():
            self.conv0 = L.Convolution2D(
                    None, 32, ksize=3, stride=1, pad=1,
                    nobias=nobias, initialW=W)
            self.bn0 = L.BatchNormalization(32)
            
            self.enc1 = Encode(32, 64)
            self.enc2 = Encode(64, 128)
            
            self.dec2 = Decode(128, 64)
            self.dec1 = Decode(64, 32)
            
            self.dconv0 = L.Convolution2D(
                    32, 3, ksize=3, stride=1, pad=1,
                    nobias=nobias, initialW=W)
    
    def __call__(self, x):
        h0 = F.relu(self.bn0(self.conv0(x)))
        h1 = self.enc1(h0)
        h = self.enc2(h1)
        h = self.dec2(h, h1)
        h = self.dec1(h, h0)
        h = self.dconv0(h)
        
        return h
        
            
    
