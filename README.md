# AnimalRecognition (matlab)
a project for computer vision        

Object: classifiy three kind of animals: dog, mountain goat and tortoise

extract Dense-SIFT features and represent images using bag of words model (with spatial tiling)
train Linear Support Vector Machines(LSVMs) classifier
visualize the performance by drawing Receiver operating characteristic (ROC) curves and Confidence Interval of True Positive Rate (TPR) and False Positive Rate (FPR) values

Positive dataset: in /data directory, dogs_all, moutaingoat_all and tortoise_all folders store about 500 images for each animal kind.
Negative dataset: in /data directory, n_all stores about 500 other animal images as negative dataset.
The original data can be randomly 80:20 splited into XXTrainI and XXTestI (400 training images and 100 test images) using randomSplitImgFile. 

The package contains 
a1dogtrain.m -- learns and test an image classifier on dog
a2goattrain.m -- learns and test an image classifier on mountaingoat
a3tortoisetrain.m -- learns and test an image classifier on tortoise
a1dogtrainRotate.m -- learns and test an image classifier on original and augmented dog images dataset

use vlfeat-0.9.20 library
computeHistogramsFromImageList.m -- compute the historgrams for list of images
computeVocabularyFromImageList.m -- compute the visual word vocabulary for list of images
cropImageSet -- crop the images in a given directory
getImageSet.m -- scan a directory for images
randomSplitImgFile -- randomly choose number train data from each path and combined into a file
rotateImageSet -- given a directory, for each image rotate it from 72 to 288 (add 4 rotated images) 
setup -- add the required search paths of vlfeat library to MATLAB
standardizeImage.m -- rescale an image with height <= 400 
trainLinearSVM.m -- Learn a linear support vector machine.
plotROC3M.m and getTPRFPRCI3.m plot the typical ROC curves and CI of TPR and FPR