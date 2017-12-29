# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 17:29:51 2017

@author: simon
"""


import numpy as np

# =============================================================================
# Provide data for the model
# =============================================================================

N=6900+1
nx=360
ny=512




def nextBatchTrain(batchSize,iteration):
    x_train=np.zeros((batchSize,nx,ny,1))  
    for i in range(batchSize):
        path = '../dataset/preProcessedSlices/Train/image{}.npy'.format(i)
        train=np.load(path)
        x_train[i,:,:,0]=np.float32(train)
    return (x_train)
           


def getTest(sizeTest):
    x_test=np.zeros((sizeTest,nx,ny,1))
    for i in range(sizeTest):
        path = '../dataset/preProcessedSlices/Test/image{}.npy'.format(i)
        testimg=np.load(path)
        x_test[i,:,:,0]=np.float32(testimg)
    return(x_test)
    
    
