function encoder = loadEncoderVeryDeep(encoderType)

if nargin < 1
  encoderType = 'vggv16-fc7' ;
end

encoder.type = encoderType ;
encoder.net = vl_simplenn_tidy(load('data/cnn/imagenet-vgg-verydeep-16.mat')) ;

switch encoderType
  case 'vggv16-conv1'
    encoder.net.layers(3:end) = [] ;
  case 'vggv16-conv2'
    encoder.net.layers(7:end) = [] ;
  case 'vggv16-conv3'
    encoder.net.layers(11:end) = [] ;
  case 'vggv16-conv4'
    encoder.net.layers(15:end) = [] ;
  case 'vggv16-conv5'
    encoder.net.layers(19:end) = [] ;
  case 'vggv16-conv6'
    encoder.net.layers(23:end) = [] ;
  case 'vggv16-conv7'
    encoder.net.layers(27:end) = [] ;
  case 'vggv16-conv8'
    encoder.net.layers(31:end) = [] ;
  case 'vggv16-conv9'
    encoder.net.layers(33:end) = [] ;
  case 'vggv16-fc7'
    encoder.net.layers(36:end) = [] ;
end

encoder.averageColor = mean(mean(encoder.net.meta.normalization.averageImage,1),2) ;
