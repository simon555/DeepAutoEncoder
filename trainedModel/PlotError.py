 # -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:40:32 2017

@author: simon
"""

import numpy as np
import matplotlib.pyplot as pl


path1='data/TrainLoss'
path2='data/TestLoss'



loss=np.loadtxt(path1)
val=np.loadtxt(path2)

fig1=pl.figure(1)
pl.plot(loss,label="Loss on Training set")
pl.plot(val,label="Loss on Test set")
pl.xlabel("epoch")
pl.ylabel('Loss function')
pl.title("Evolution of the loss functions, \n after a pretraining on 600 images")
pl.axis([0,100,0,0.003])
pl.legend()

pl.savefig('errorEvolution.png')
pl.show()


