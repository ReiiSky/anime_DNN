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
        
            
class UNET_02(chainer.Chain):
    
    def __init__(self):
        super(UNET_02, self).__init__()
        
        W = chainer.initializers.HeNormal()
        nobias = True
        
        with self.init_scope():
            self.c0 = L.Convolution2D(
                    None, 32, ksize=3, stride=1, pad=1,
                    nobias=nobias, initialW=W)
            self.bn0 = L.BatchNormalization(32)
            
            self.c1 = L.Convolution2D(
                    32, 64, ksize=4, stride=2, pad=1,
                    nobias=nobias, initialW=W)
            self.bn1 = L.BatchNormalization(64)
            
            self.c2 = L.Convolution2D(
                    64, 64, ksize=3, stride=1, pad=1,
                    nobias=nobias, initialW=W)
            self.bn2 = L.BatchNormalization(64)
            
            self.c3 = L.Convolution2D(
                    64, 128, ksize=4, stride=2, pad=1,
                    nobias=nobias, initialW=W)
            self.bn3 = L.BatchNormalization(128)
            
            self.c4 = L.Convolution2D(
                    128, 128, ksize=3, stride=1, pad=1,
                    nobias=nobias, initialW=W)
            self.bn4 = L.BatchNormalization(128)
            
            self.c5 = L.Convolution2D(
                    128, 256, ksize=4, stride=2, pad=1,
                    nobias=nobias, initialW=W)
            self.bn5 = L.BatchNormalization(256)
            
            self.c6 = L.Convolution2D(
                    256, 256, ksize=3, stride=1, pad=1,
                    nobias=nobias, initialW=W)
            self.bn6 = L.BatchNormalization(256)
            
            self.dc6 = L.Convolution2D(
                    256, 256, ksize=3, stride=1, pad=1,
                    nobias=nobias, initialW=W)
            self.dbn6 = L.BatchNormalization(256)
            
            self.dc5 = L.Deconvolution2D(
                    256, 128, ksize=4, stride=2, pad=1,
                    nobias=nobias, initialW=W)
            self.dbn5 = L.BatchNormalization(128)
            
            self.dc4 = L.Convolution2D(
                    128, 128, ksize=3, stride=1, pad=1,
                    nobias=nobias, initialW=W)
            self.dbn4 = L.BatchNormalization(128)
            
            self.dc3 = L.Deconvolution2D(
                    128, 64, ksize=4, stride=2, pad=1,
                    nobias=nobias, initialW=W)
            self.dbn3 = L.BatchNormalization(64)
            
            self.dc2 = L.Convolution2D(
                    64, 64, ksize=3, stride=1, pad=1,
                    nobias=nobias, initialW=W)
            self.dbn2 = L.BatchNormalization(64)
            
            self.dc1 = L.Deconvolution2D(
                    64, 32, ksize=4, stride=2, pad=1,
                    nobias=nobias, initialW=W)
            self.dbn1 = L.BatchNormalization(32)
            
            self.dc0 = L.Convolution2D(
                    32, 3, ksize=3, stride=1, pad=1,
                    nobias=nobias, initialW=W)
            
    def __call__(self, x):
        h0 = F.relu(self.bn0(self.c0(x)))
        h = F.relu(self.bn1(self.c1(h0)))
        h2 = F.relu(self.bn2(self.c2(h)))
        h = F.relu(self.bn3(self.c3(h2)))
        h4 = F.relu(self.bn4(self.c4(h)))
        h5 = F.relu(self.bn5(self.c5(h4)))
        h6 = F.relu(self.bn6(self.c6(h5)))
        
        dh = F.relu(self.dbn6(self.dc6(h5 + h6)))
        dh = F.relu(self.dbn5(self.dc5(dh)))
        dh = F.relu(self.dbn4(self.dc4(dh + h4)))
        dh = F.relu(self.dbn3(self.dc3(dh)))
        dh = F.relu(self.dbn2(self.dc2(dh + h2)))
        dh = F.relu(self.dbn1(self.dc1(dh)))
        dh = self.dc0(dh + h0)
        
        return dh
