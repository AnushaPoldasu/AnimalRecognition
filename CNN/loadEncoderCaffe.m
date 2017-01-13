function encoder = loadEncoderCaffe(encoderType)

if nargin < 1
  encoderType = 'caffe-fc7' ;
end

encoder.type = encoderType ;
encoder.net = vl_simplenn_tidy(load('data/cnn/imagenet-caffe-ref.mat')) ;

switch encoderType
  case 'caffe-conv1'
    encoder.net.layers(3:end) = [] ;
  case 'caffe-conv2'
    encoder.net.layers(7:end) = [] ;
  case 'caffe-conv3'
    encoder.net.layers(11:end) = [] ;
  case 'caffe-conv4'
    encoder.net.layers(13:end) = [] ;
  case 'caffe-conv5'
    encoder.net.layers(15:end) = [] ;
  case 'caffe-fc7'
    encoder.net.layers(20:end) = [] ;
end

encoder.averageColor = mean(mean(encoder.net.meta.normalization.averageImage,1),2) ;
