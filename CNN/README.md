# AnimalRecognition-CNN

The package contains 
a1dogtrain.m -- learns and test an image classifier on dog
a2goattrain.m -- learns and test an image classifier on mountaingoat
a3tortoisetrain.m -- learns and test an image classifier on tortoise
a1dogtrainRotate.m -- learns and test an image classifier on original and augmented dog images dataset

based on VLFeat and MatConvNet 

computediscFromImageList.m -- encode list of images using cnn features  
getImageSet.m -- scan a directory for images
loadEncodervgg128.m: Load the encoder imagenet-vgg-m-128.mat
loadEncoderCaffe.m: Load the encoder imagenet-caffe-ref.mat
loadEncoderVeryDeep.m: Load the encoder imagenet-vgg-verydeep-16.mat
randomSplitImgFile.m -- randomly choose number train data from each path and combined into a file
rotateImageSet.m -- given a directory, for each image rotate it from 72 to 288 (add 4 rotated images) 
setup.m -- add the required search paths of vlfeat library to MATLAB
standardizeImage.m -- consider the filter size, we rescale every image <= 256 * 256 
trainLinearSVM.m -- Learn a linear support vector machine.
plotROCrotateCNN.m, plotROC3M.m and getTPRFPRCI3.m plot the typical ROC curves and CI of TPR and FPR

The networks are already pretrained on the ImageNet dataset. (available at http://www.vlfeat.org/matconvnet/pretrained/) 
download imagenet-vgg-m-128.mat, imagenet-caffe-ref.mat and imagenet-vgg-verydeep-16.mat (or the dataset you want to try) and save them in data/cnn
