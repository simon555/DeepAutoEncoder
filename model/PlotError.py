 # -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:40:32 2017

@author: simon
"""

import numpy as np
import matplotlib.pyplot as pl
import os



def plot_results(Nexp):
    directory='./resultsOfTraining/Exp{}/data/'.format(Nexp)
    
    print('results of experience ',Nexp)
    
    if not os.path.exists(directory):
        print('ERROR')
        print('folder {} does not exist'.format(directory))
        print('are you trying to plot the right experience?')
        return(False)
        
    pathTrain=directory+'TrainLoss'
    pathTest=directory+'TestLoss'
    if not (os.path.exists(pathTrain) and os.path.exists(pathTest)):
        print('Error')
        print('data not available, check your folders')
        return(False)
    
    train=np.loadtxt(pathTrain)
    test=np.loadtxt(pathTest)
    
    x_max=len(train)
    y_min=min(np.min(train),np.min(test))
    y_max=max(np.max(train),np.max(test))
    
    
    
    fig1=pl.figure(1)
    pl.plot(train,label="Loss on Training set")
    pl.plot(test,label="Loss on Test set")
    pl.xlabel("epoch")
    pl.ylabel('Loss function')
    pl.title("Evolution of the loss functions")
    pl.axis([0,x_max,y_min,y_max])
    pl.legend()
    
    pl.savefig(directory+'errorEvolution.png')
    pl.show()


