# Animal Recognition (Matlab)

Classifiy images of three kind of animals: dog, mountain goat and tortoise 

extract Dense-SIFT features and represent images using bag of words model (with spatial tiling) and three Covolutional-Neural-Networking encoders.

train Linear Support Vector Machines(LSVMs) classifier

visualize the performance by drawing Receiver operating characteristic (ROC) curves and Confidence Interval of True Positive Rate (TPR) and False Positive Rate (FPR) values


Dataset
Positive dataset: in /data directory, dogs_all, moutaingoat_all and tortoise_all folders store about 500 images for each animal kind.

Negative dataset: in /data directory, n_all stores about 500 other animal images as negative dataset.

The original data can be randomly 80:20 splited into XXTrainI and XXTestI (400 training images and 100 test images) using randomSplitImgFile. 



