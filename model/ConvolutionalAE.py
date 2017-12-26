# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 16:09:00 2017

@author: simon
"""

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, merge, ZeroPadding2D,BatchNormalization,Flatten, Activation
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard,CSVLogger
from keras.models import load_model
import keras
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as pl 
from keras import regularizers
import provider
#from multiple_gpu import make_parallel
from keras.layers.core import Lambda,Reshape
from keras.layers.advanced_activations import PReLU
import os
K.clear_session()


NumeroArchitecture=4

dropout=0.2
    
nx=360
ny=512
Nepochs=100
Nsamples=4
sizeTest=4
batchSize=4

filterSize=3

NumberOfMachines=1

Nbatches=int(Nsamples//batchSize)
    
print("loading the data...")

x_train=provider.nextBatchTrain(Nsamples,0)
x_test=provider.getTest(sizeTest)


Nexp=1
directory='./resultsOfTraining/Exp{}/'.format(Nexp)

if not os.path.exists(directory):
    print('new directory : ',directory)
    
else:
    while(os.path.exists(directory)):
        print('directory already exists : ',directory)
        Nexp+=1
        directory='resultsOfTraining/Exp{}/'.format(Nexp)
    print('new directory : ',directory)
        
directoryData=directory+'data/'
directoryModel=directory+'models/'
directoryImages=directory+'images/'


os.makedirs(directory) 
os.makedirs(directoryData)
os.makedirs(directoryModel)
os.makedirs(directoryImages)


csv_logger = CSVLogger(directory+'historic.log')


Nprint=1
Nsave=5

class My_Callback(keras.callbacks.Callback):
    def __init__(self, Nprint, Nsave):
        self.Nprint = Nprint
        self.Nsave=Nsave
        self.batch = 0
        self.epoch=1
      
           
    def on_epoch_end(self,batch,epoch,logs={}):
        #dropout/=2
        fig=pl.figure(1)
        x_t=x_test[0:4,...]
        y_t=autoencoder.predict(x_t)

        x_img=x_t[0,:,:,0]
        y_img=y_t[0,:,:,0]
        error=np.abs(x_img-y_img)

        fig=pl.figure(1)
        
        ax1=fig.add_subplot(221)
        pl.title("original")
        ax1.imshow(x_img,cmap='jet')

        ax2=fig.add_subplot(222)
        pl.title('reconstructed')
        ax2.imshow(y_img,cmap='jet')

        ax3=fig.add_subplot(224)
        pl.title('error')
        ax3.imshow(error,cmap="jet")

        pl.legend()
        pl.savefig(directoryImages+"image_epoch{}".format(self.epoch))
        
        
        if self.epoch % self.Nsave == 0:
            autoencoder.save(directoryModel+'model_epoch{}.h5'.format(self.epoch))  # creates a HDF5 file 'my_model.h5'
        self.epoch+=1

        
myCallback=My_Callback(Nprint,Nsave)
    
    
print("defining the model...")

input_img = Input(shape=(nx, ny, 1))  # adapt this if using `channels_first` image data format
#(512,360)



#x=BatchNormalization()(x)
#x=Dropout(dropout)(x)
x = Conv2D(10, (filterSize, filterSize), padding='same')(input_img)    
x=BatchNormalization()(x)
x=Activation("relu")(x)


x = Conv2D(20, (filterSize, filterSize), padding='same')(x)    
x=BatchNormalization()(x)
x=Activation("relu")(x)

x = MaxPooling2D((2, 2), padding='same')(x)
x=Conv2D(40, (filterSize, filterSize), padding='same')(x)    
x=Activation("relu")(x)
y=Conv2D(40, (filterSize, filterSize), padding='same')(x)  
y=BatchNormalization()(y)
y=Activation("relu")(y)
#256,180
s3 = keras.layers.Add()([x, y])


x = MaxPooling2D((2, 2), padding='same')(s3)                           
x=Conv2D(40, (filterSize, filterSize), padding='same')(x)    
x=Activation("relu")(x)
y=Conv2D(40, (filterSize, filterSize), padding='same')(x)  
y=BatchNormalization()(y)
y=Activation("relu")(y)
#128,90
s2 = keras.layers.Add()([x, y])


x = MaxPooling2D((2, 2), padding='same')(s2)   
x=Conv2D(40, (filterSize, filterSize), padding='same')(x)    
x=Activation("relu")(x)
y=Conv2D(40, (filterSize, filterSize), padding='same')(x)  
y=BatchNormalization()(y)
y=Activation("relu")(y)
#64,45
s1 = keras.layers.Add()([x, y])

x = Conv2D(80, (filterSize, filterSize), padding='same')(s1)    
x=BatchNormalization()(x)
x=Activation("relu")(x)



#HERE IS THE CODE
encoded=x



#START DECODING

x = Conv2D(80, (filterSize, filterSize), padding='same')(encoded) 
x=BatchNormalization()(x)
x=Activation("relu")(x)


x = Conv2D(40, (filterSize, filterSize), padding='same')(x) 
x=Activation("relu")(x)
x = Conv2D(40, (filterSize, filterSize), padding='same')(x) 
x=BatchNormalization()(x)
x=Activation("relu")(x)
x = keras.layers.Add()([x, s1])

x = UpSampling2D((2, 2))(x) 
#128,90
x = Conv2D(40, (filterSize, filterSize), padding='same')(x) 
x=Activation("relu")(x)
x = Conv2D(40, (filterSize, filterSize), padding='same')(x) 
x=BatchNormalization()(x)
x=Activation("relu")(x)
x = keras.layers.Add()([x, s2])

x = UpSampling2D((2, 2))(x) 
x = Conv2D(40, (filterSize, filterSize), padding='same')(x) 
x=Activation("relu")(x)
x = Conv2D(40, (filterSize, filterSize), padding='same')(x) 
x=BatchNormalization()(x)
x=Activation("relu")(x)
x = keras.layers.Add()([x, s3])


x = UpSampling2D((2, 2))(x) 
x = Conv2D(40, (filterSize, filterSize), padding='same')(x)
x=Activation("relu")(x) 
x = Conv2D(30, (filterSize, filterSize), padding='same')(x)
x=Activation("relu")(x)
x = Conv2D(20, (filterSize, filterSize), padding='same')(x)
x=Activation("relu")(x)
x = Conv2D(10, (filterSize, filterSize), padding='same')(x)
x=Activation("relu")(x) 
x = Conv2D(1, (filterSize, filterSize), padding='same')(x)
x=Activation("relu")(x) 
decoded=x




autoencoder=Model(input_img,decoded)

print("parallelizing the model...", end='')
if NumberOfMachines>1:
    #autoencoder=make_parallel(autoencoder,NumberOfMachines)
    print('on {} machines'.format(NumberOfMachines))
    print('Need to get back make_parallel.py from the server!')
else:
    print("False")





   
print("model set up")

autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

print("model compiled")

print("beginning of the training")
hist=autoencoder.fit(x_train, x_train,
                epochs=Nepochs,
                batch_size=batchSize,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[myCallback,csv_logger])
                
print(autoencoder.summary())
          

print('training finished')

print('saving the model')
autoencoder.save(directoryModel+'my_modelFinal.h5')# creates a HDF5 file 'my_model.h5'


print('saving the data stats')
hLoss=hist.history['loss']
hValidation=hist.history['val_loss']
np.savetxt(directoryData+"TrainLoss",hLoss)
np.savetxt(directoryData+"TestLoss",hValidation)

print("end")
