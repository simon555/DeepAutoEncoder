# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 16:09:00 2017

@author: simon
"""

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, merge, ZeroPadding2D,BatchNormalization,Flatten, Activation
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard,CSVLogger
from keras.models import load_model
import keras
import matplotlib.pyplot as pl
from keras import regularizers
import providerKerasLinux as provider
#from multiple_gpu import make_parallel
from keras.layers.core import Lambda,Reshape
from keras.layers.advanced_activations import PReLU

K.clear_session()


NumeroArchitecture=4

dropout=0.2
    
nx=360
ny=512
Nepochs=100
Nsamples=600
sizeTest=40
batchSize=4

filterSize=3

Nbatches=int(Nsamples//batchSize)
    
print("loading the data...")

x_train=provider.nextBatchTrain(Nsamples,0)
x_test=provider.getTest(sizeTest)


#test
#x_train=x_train[0:1000,...]


#x_train = x_train.astype('float32') 
#x_test = x_test.astype('float32') 
#x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
#x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

csv_logger = CSVLogger('/workspace/newExp/results/A{}/scripts/historic.log'.format(NumeroArchitecture))


Nprint=1
Nsave=5

class My_Callback(keras.callbacks.Callback):
    def __init__(self, Nprint, Nsave):
        self.Nprint = Nprint
        self.Nsave=Nsave
        self.batch = 0
        self.epoch=1
      
    #def on_batch_end(self, batch, logs={}):
     #   if self.batch % self.Nprint == 0:
      #      fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8,5))
       #     x_test=x_train[0:1,...]
        #    result=autoencoder.predict(x_test)
         #   ax[0].imshow(x_test[0,...,0], cmap='jet')
          #  ax[1].imshow(result[0,...,0], cmap='jet')
           # ax[0].set_title("Input")
            #ax[1].set_title("decode - code")
            #fig.tight_layout()
            #pl.show()
       # self.batch += 1
        
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
        pl.savefig("/workspace/newExp/results/A{}/imageResult/image_A{}_epoch{}".format(NumeroArchitecture,NumeroArchitecture,self.epoch))
        
        
        if self.epoch % self.Nsave == 0:
            autoencoder.save('/workspace/newExp/results/A{}/modelResult/model_A{}_epoch{}.h5'.format(NumeroArchitecture,NumeroArchitecture,self.epoch))  # creates a HDF5 file 'my_model.h5'
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
x = Conv2D(20, (filterSize, filterSize), padding='same')(x)
x=Activation("relu")(x) 
x = Conv2D(10, (filterSize, filterSize), padding='same')(x)
x=Activation("relu")(x)
x = Conv2D(1, (filterSize, filterSize), padding='same')(x)
x=Activation("relu")(x)
#decoded=BatchNormalization()(x)
decoded=x



print("parallelizing the model")
autoencoder=Model(input_img,decoded)
#autoencoder=make_parallel(autoencoder,4)




#L=autoencoder.layers
#l1=MaxPooling2D((2, 2), padding='valid')
#l2=Conv2D(40, (filterSize, filterSize), activation='relu', padding='same')
#l3=Conv2D(40, (filterSize, filterSize), activation='relu', padding='same')
#l4=UpSampling2D((2, 2))
#l5=ZeroPadding2D( padding= ((1,0),(0,0)) )
#autoencoder.layers=L[0:6]+[l1,l2,l3,l4,l5]+L[6:]
    
#print("I reload the model...")
#path='/root/scripts/keras/results/A10/modelResult/model_A10_epoch20.h5'
#autoencoder=load_model(path) 
#autoencoder=autoencoder.layers[-2]


   
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
                
print(autoencoder.layers)
                
#for epoch in range(Nepochs):
#    for iteration in range(Nbatches):
#        print("beginning of the iteration {} in epoch {}".format(iteration,epoch))
#        x_train=provider.nextBatchTrain(batchSize,iteration)
#        autoencoder.train_on_batch(x_train, x_train)
#        print("end of the iteration {} in epoch {}".format(iteration,epoch))
#       autoencoder.fit(x_train,x_train,epochs=1,batch_size=batchSize,validation_data=(x_test, x_test),callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
#
 #   print('epoch {}'.format(epoch))
 #   if epoch%Nsave==0:
 #       autoencoder.save('my_model{}.h5'.format(epoch))


#decoded_imgs = autoencoder.predict(x_test)
#
#n = 10
#plt.figure(figsize=(20, 4))
#for i in range(n):
#    # display original
#    ax = plt.subplot(2, n, i+1)
#    plt.imshow(x_test[i].reshape(28, 28))
#    plt.gray()
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#
#    # display reconstruction
#    ax = plt.subplot(2, n, i + n+1)
#    plt.imshow(decoded_imgs[i].reshape(28, 28))
#    plt.gray()
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#plt.show()

print('training finished')

print('saving the model')
#autoencoder.save('/root/workspace/results/A{}/modelResult/my_modelFinal.h5'.format(NumeroArchitecture))  # creates a HDF5 file 'my_model.h5'


print('saving the data stats')
hLoss=hist.history['loss']
hValidation=hist.history['val_loss']
np.savetxt("/workspace/newExp/results/A{}/scripts/A{}_Loss".format(NumeroArchitecture,NumeroArchitecture),hLoss)
np.savetxt("/workspace/newExp/results/A{}/scripts/A{}_ValidationLoss".format(NumeroArchitecture,NumeroArchitecture),hValidation)

print("end")
