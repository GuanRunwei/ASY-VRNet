# Efficient-VRNet: An Exquisite Fusion Network for Riverway Panoptic Perception based on Asymmetric Fair Fusion of Vision and 4D mmWave Radar

# Device:
Camera: Sony IMX-317

Radar: Ocuii Imaging Radar

# Implementation:

* Create a conda environment and install dependencies
> git clone https://github.com/GuanRunwei/Efficient-VRNet.git \
> cd Efficient-VRNet   \
> conda create -n efficientvrnet python=3.7  \
> conda activate efficientvrnet   \
> pip install -r requirements.txt


* Prepare datasets for object detection and semantic segmentation based on image and radar

> For object detection, make two files, one for training and another for test. There are two ways to complete this. \
> 1. Make two txt files, one for training and another for test. Two files are with the same format: 
> In one line, there are two parts, an image path and objects:  
> **image path**: E:/Big_Datasets/water_surface/all-1114/all/VOCdevkit/VOC2007/JPEGImages/1664091257.87023.jpg  
> **object 1** (the first four numbers are the bounding box and the last is the category): 1131,430,1152,473,0   
> **object 2**: 920,425,937,451,0   
> Therefore, each line is like this: E:/Big_Datasets/water_surface/all-1114/all/VOCdevkit/VOC2007/JPEGImages/1664091257.87023.jpg 1131,430,1152,473,0 920,425,937,451,0 
> 2. Organize the files in VOC format in one folder like this: \
> VOCdevkit \
> -VOC2007  \
> -- Annotations -> xml annotations in VOC format (you need to put annotations in it)  \
> -- ImageSets -> id (you do not need to do)  \
> -- JPEGImages  -> images (you need to put images in it)  \
> **enter** voc_annotation.py and follow the annotation to make your dataset


> For semantic segmentation, make the folders in VOC format. \
> VOCdevkit \
> -VOC2007  \
> -- ImageSets -> id (you do not need to do)  \
> -- JPEGImages  -> images (you need to put images in it)  \
> -- SegmentationClass -> images_seg (you need to put images in it)  


> For radar files, you need to make the radar map with the spatial size of images for object detection. 
We need four features: range, velocity, elevation and power, so firstly project 3D point clouds into 2D image plane,
then make a numpy matrix with 4×512×512, each channel means one feature. Then, save the numpy matrix in npz format.

> ***Attention:*** The names of images for object detection,segmentation, and radar must be the same!
The only difference between them is the format.

* Train

> After you have completed the above, enter the **train.py**. Change the file path variables and hyperparameters and run it.


* Visualization

> **predict.py** is to test the object detection  \
> **yolo.py** is to define the model for object detection \

> **predict_seg.py** is to test the semantic segmentation  \
> **deeplab.py** is to define the model for semantic segmentation \

> enter these files see the details by annotations in the files

If have any questions, put them in Issues ---

