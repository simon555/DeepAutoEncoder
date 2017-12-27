# DeepAutoEncoder

In this project, our goal is to build an autoencoder in order to improve state-of-the-art image registration algorithm. This algorithm uses subsampled versions of images to register in order to compute a deformation flow, but we expect that by using more wise features we could improve the results.
In this part of the work, we built a convolutional autoencoder that is able to retrieve the information contained in an image, by producing multiple image features of low resolution.

Below an outline of the architecture used : 
![](https://github.com/simon555/DeepAutoEncoder/blob/master/trainedModel/ArchitectureDiagram.png)

To train this model, we first used a sample of our dataset composed of 6000 images. Using only 600 of them can allow us to get a good accuracy in a short time (~15min). *Need access to the server to get the plot of the error on pretraining*.
Then, we train the model on the full dataset :

![](https://github.com/simon555/DeepAutoEncoder/blob/master/trainedModel/errorEvolution.png)

