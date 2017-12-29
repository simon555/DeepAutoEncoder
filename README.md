# Requirements
* Python 3+
* TensorFlow
* Keras
* Unrar


## You should first dowload the dataset ##
To download the dataset, download the files on my private google drive and copy them into dataset/raw3DVolumes/   
To be able to download, you should have access to the drive and be connected on your account.
```
https://drive.google.com/uc?export=download&id=17QRHX5JSmqliQv_x6S1yUMhk_Sx5TqvI
```
copy the data-mvct.rar in the folder dataset/ and decompresses it into the folder raw3DVolumes (automatically created with the following code)
```
unrar e data-mvct.rar raw3DVolumes/
rm data-mvct.rar
```
You can now pre process these raw 3D volumes with the python script preProcessData.py that will generate the exploitable dataset.






# DeepAutoEncoder

In this project, our goal is to build an autoencoder in order to improve state-of-the-art image registration algorithm. This algorithm uses subsampled versions of images to register in order to compute a deformation flow, but we expect that by using more wise features we could improve the results.
In this part of the work, we built a convolutional autoencoder that is able to retrieve the information contained in an image, by producing multiple image features of low resolution.

Below an outline of the architecture used : 
![](https://github.com/simon555/DeepAutoEncoder/blob/master/trainedModel/ArchitectureDiagram.png)

To train this model, we first used a sample of our dataset composed of 6000 images. Using only 600 of them can allow us to get a good accuracy in a short time (~15min). *Need access to the server to get the plot of the error on pretraining*.
Then, we train the model on the full dataset :

![](https://github.com/simon555/DeepAutoEncoder/blob/master/trainedModel/errorEvolution.png)

Here is an exemple of the reconstrution, with the fully trained model : 
![](https://github.com/simon555/DeepAutoEncoder/blob/master/trainedModel/images/finalResult.png)
