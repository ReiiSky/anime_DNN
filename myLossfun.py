# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 20:26:57 2018

@author: taishi-o
"""

import numpy as np

from chainer import function_node
from chainer.backends import cuda
import chainer.functions
from chainer.utils import type_check


class L1L2NormError(function_node.FunctionNode):
    
    def __init__(self, lam):
        self.lam = lam
    
    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == np.float32,
            in_types[1].dtype == np.float32,
            in_types[0].shape == in_types[1].shape
            )
        
    def forward(self, inputs):
        self.retain_inputs((0, 1))
        xp = cuda.get_array_module(*inputs)
        
        diff = inputs[0] - inputs[1]
        self.sign = xp.sign(diff)
        diff = diff.ravel()
        
        return (diff.dot(diff) + self.lam*xp.sum(xp.abs(diff))) / diff.dtype.type(diff.size) ,
    
    def backward(self, indexes, gy):
        x, t = self.get_retained_inputs()
        diff = x - t
        gy0 = chainer.functions.broadcast_to(gy[0], diff.shape)
        gx = gy0 * (2 * diff + self.lam * self.sign) / diff.size
        
        return gx, -gx
    

def l1l2_norm_error(x, t, lam):
    
    return L1L2NormError(lam = lam).apply((x, t))[0]
        
