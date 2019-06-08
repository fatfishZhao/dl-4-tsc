from __future__ import division
import cv2
import numpy as np
from numpy import random
import math

class array_to3d(object):
    def __init__(self, mode):
        self.model = mode
    def __call__(self, data):
        '''
        :param data: length * fea_num
        '''
        # transform order: 12345613514624

        data = data[:,:,np.newaxis]
        data = np.concatenate((data,data,data), axis=2)
        return data
class array_transpose(object):
    def __init__(self,mode):
        self.mode = mode
    def __call__(self, data):
        '''
        :param data: length * fea_num * 3
        '''
        data = data.transpose((2,0,1))
        return data