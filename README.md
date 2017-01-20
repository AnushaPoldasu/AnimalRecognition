# Animal Recognition (Matlab)

The project is to classify images of three kinds of animals.

The first method is to extract the Dense-SIFT features, and represent images using bag of words model (with spatial tiling), and then train Linear Support Vector Machines(LSVMs) classifier.

To improve the performance, in the second approach, I used three Deep Convolutional Neural Networks (CNN) encoders to learn features of images and trained SVMs for classification.

The performance is visualized drawing Receiver operating characteristic (ROC) curves and Confidence Interval of True Positive Rate (TPR) and False Positive Rate (FPR) values


Dataset is in /data directory:
Positive dataset: dogs_all, moutaingoat_all and tortoise_all folders store about 500 images for each animal kind.

Negative dataset: n_all stores about 500 other animal images as negative dataset.

The original data can be randomly 80:20 splited into AnimalNameTrainI and AnimalNameTestI (400 training images and 100 test images) using randomSplitImgFile. 



