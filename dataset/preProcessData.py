# -*- coding: utf-8 -*-

#Created on Mon Jun 19 11:37:38 2017

#@author: simon
###

import skimage.io as io
import numpy as np
import os

directory='preProcessedSlices/'
directoryTrain=directory+'Train/'
directoryTest=directory+'Test/'

if not os.path.exists(directory):
    print('new directory : ',directory)
    os.makedirs(directory)
    os.makedirs(directoryTest)
    os.makedirs(directoryTrain)
else:
    print('directory already exists')

#==============================================================================
# go through the images 
#==============================================================================

#AXIS Z

    
def process_data(data):
    # normalization      
    data=data[70:430,:]
    data[data<-600]=-1024    
    data=np.clip(data,-np.inf,1024)	


    mi=np.min(data)
    ma=np.max(data)
    data=(data-mi)/(ma-mi)
     
    #m=np.mean(data)
    #st=np.std(data)
    #data=(data-m)/st
    
    return (data)
    
counter=0
labelImage=0

print('processing the Train set')
for patient in range (11,16):
    for image in range(33):
        if image<10:
            path='raw3DVolumes/patient{}_mvct_0{}.mha'.format(patient,image)
        else:
            path='raw3DVolumes/patient{}_mvct_{}.mha'.format(patient,image)
        
        try:
            img= io.imread(path,plugin='simpleitk')
            a,b,c=img.shape
            for coupe in range(a):
                if (counter<2907 or counter>2974):
                    imageTranche=img[coupe,:,:]
                    imageModif=process_data(imageTranche)
                    np.save(directoryTrain+"image{}".format(labelImage),imageModif)
                    labelImage+=1
                    if labelImage%100==0:
                        print(labelImage)
                        print('patient {}, image {} coupe : {}'.format(patient,image,coupe))
                else:
                    print("bad image, counter {}".format(counter))
                counter+=1
                       
        except:
            print('patient {}, image {} inexistante'.format(patient, image))
    
    
    
    
counter=0
labelImage=0

print('processing the Test set')
for patient in range (16,17):
    for image in range(33):
        if image<10:
            path='raw3DVolumes/patient{}_mvct_0{}.mha'.format(patient,image)
        else:
            path='raw3DVolumes/patient{}_mvct_{}.mha'.format(patient,image)
        
        try:
            img= io.imread(path,plugin='simpleitk')
            a,b,c=img.shape
            for coupe in range(a):
                if (counter<2907 or counter>2974):
                    imageTranche=img[coupe,:,:]
                    imageModif=process_data(imageTranche)
                    np.save(directoryTest+"image{}".format(labelImage),imageModif)
                    labelImage+=1
                    if labelImage%100==0:
                        print(labelImage)
                        print('patient {}, image {} coupe : {}'.format(patient,image,coupe))
                else:
                    print("bad image, counter {}".format(counter))
                counter+=1
                       
        except:
            print('patient {}, image {} inexistante'.format(patient, image))
      
  