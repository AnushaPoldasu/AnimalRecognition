% --------------------------------------------------------------------
% train the tortoise dataset
% --------------------------------------------------------------------
setup ;
% For stage I: Change the CNN representation
% Change the CNN representation
%encoding = 'vggm128-conv1';
%encoding = 'vggm128-conv2';
% encoding = 'vggm128-conv3';
%encoding = 'vggm128-conv4';
%encoding = 'vggm128-conv5';
% encoding = 'vggm128-fc7';
% encoder = loadEncoder(encoding);

% Change the CNN representation
%encoding = 'caffe-conv1';
%encoding = 'caffe-conv2';
% encoding = 'caffe-conv3';
%encoding = 'caffe-conv4';
%encoding = 'caffe-conv5';
% encoding = 'caffe-fc7';
% encoder = loadEncoderCaffe(encoding);
% Change the CNN representation
%encoding = 'vggv16-conv1';
%encoding = 'vggv16-conv2';
% encoding = 'vggv16-conv3';
%encoding = 'vggv16-conv4';
%encoding = 'vggv16-conv5';
%encoding = 'vggv16-conv6';
% encoding = 'vggv16-conv7';
%encoding = 'vggv16-conv8';
%encoding = 'vggv16-conv9';
encoding = 'vggv16-fc7' ;
encoder = loadEncoderVeryDeep(encoding);

T = cell(1,100);
Tpr = cell(1,100);
Fpr = cell(1,100);
Score = cell(1,100);
for z = 1:100
    % --------------------------------------------------------------------
    % Stage I: Data Preparation
    % --------------------------------------------------------------------
%     % step 1 get the splitted training and testing images
%     % positive data
%     fprintf('%s\n','slpit files');
%     spath = 'data/tortoise_all';
%     dpath1 = 'data/tortoiseTrainI';
%     dpath2 = 'data/tortoiseTestI';
%     randomSplitImgFile(spath, dpath1,dpath2,100); % spath copy to dpath1, dpath1 move 100 to dpath2
%     
%     % negative dataspath = 'data/n_all';
%     dpath1 = 'data/nTrainIt';
%     dpath2 = 'data/nTestIt';
%     randomSplitImgFile(spath, dpath1,dpath2,100);
%     %----------------------------------------------------------------------
%     % step 2 generate histograms for tortoise and negative samples
%     % tortoise test
%     path = 'data/tortoiseTestI';
%     tortoisetest_names = getImageSet(path);
%     tortoisetest_descriptors = computepsisFromImageList(encoder,tortoisetest_names);
%     % tortoise train
%     path = 'data/tortoiseTrainI';
%     tortoisetrain_names = getImageSet(path);
%     tortoisetrain_descriptors = computepsisFromImageList(encoder, tortoisetrain_names);
%     % ntest
%     path = 'data/nTestIt';
%     ntestt_names = getImageSet(path);
%     ntestt_descriptors = computepsisFromImageList(encoder,ntestt_names);
%     % ntrain
%     path = 'data/nTrainIt';
%     ntraint_names = getImageSet(path);
%     ntraint_descriptors = computepsisFromImageList(encoder,ntraint_names);
%     %----------------------------------------------------------------------
%     % step 3  save the descriptors and names
%     filename5='data/tortoise_train_psis';
%     save(filename5,'tortoisetrain_descriptors','tortoisetrain_names');
%     filename6='data/tortoise_test_psis';
%     save(filename6,'tortoisetest_descriptors','tortoisetest_names');
%     filename7='data/n_traint_psis';
%     save(filename7,'ntraint_descriptors','ntraint_names');
%     filename8='data/n_testt_psis';
%     save(filename8,'ntestt_descriptors','ntestt_names');
    %----------------------------------------------------------------------
    % step 4
    % Load training data
    numPos = +inf;
    numNeg = +inf;
    names = {};
    descriptors = [];
    labels = [];
    pos = load('data/tortoise_train_psis.mat');
    neg = load('data/n_traint_psis.mat');
    selp = vl_colsubset(1:numel(pos.tortoisetrain_names),numPos,'beginning');
    seln = vl_colsubset(1:numel(neg.ntraint_names),numNeg,'beginning');
    names = horzcat(names, pos.tortoisetrain_names(selp), neg.ntraint_names(seln));
    descriptors = horzcat(descriptors, pos.tortoisetrain_descriptors(:,selp), neg.ntraint_descriptors(:,seln));
    labels = horzcat(labels, ones(1,numel(selp)), - ones(1,numel(seln)));
    clear pos neg;

    % Load testing data
    testNames = {} ;
    testDescriptors = [] ;
    testLabels = [] ;
    pos = load('data/tortoise_test_psis') ;
    neg = load('data/n_testt_psis.mat') ;
    selp = vl_colsubset(1:numel(pos.tortoisetest_names),numPos,'beginning');
    seln = vl_colsubset(1:numel(neg.ntestt_names),numNeg,'beginning');
    testNames = {pos.tortoisetest_names{:}, neg.ntestt_names{:}};
    testLabels = [ones(1,numel(pos.tortoisetest_names)), - ones(1,numel(neg.ntestt_names))];
    testDescriptors = cat(3, testDescriptors, [pos.tortoisetest_descriptors, neg.ntestt_descriptors]);
    clear pos neg;

    % count how many images are there
    fprintf('Number of training images: %d positive, %d negative\n', sum(labels > 0), sum(labels < 0));
    fprintf('Number of testing images: %d positive, %d negative\n', sum(testLabels > 0), sum(testLabels < 0));

    % normalize the histograms before running the linear SVM
    descriptors = bsxfun(@times, descriptors, 1./sqrt(sum(descriptors.^2,1)));
    testDescriptors = bsxfun(@times, testDescriptors, 1./sqrt(sum(testDescriptors.^2,1)));

    % --------------------------------------------------------------------
    % Stage II: Training a classifier 
    % --------------------------------------------------------------------
    C = 10 ;
    [w, bias] = trainLinearSVM(descriptors, labels, C);
    % --------------------------------------------------------------------
    % Stage III: Classify the test images and assess the performance
    % --------------------------------------------------------------------
    % Test the linar SVM
    testScores = w' * testDescriptors + bias;
    % --------------------------------------------------------------------
    % Visualize the roc curve
    figure(1) ; clf ; set(1,'name','Receiver Operating Characteristic on test data');
    vl_roc(testLabels, testScores);
    % Print results
    % --------------------------------------------------------------------
    [tpr, tnr, info2] = vl_roc(testLabels, testScores);
    fprintf('Test ROC: %.3f\n', info2.auc);
    s = 'rocTortoise';
    s1= strcat(s,sprintf('%d%s',z,'.fig'));
%   fullPath = fullfile('data/','tortoiseROCs1vgg',s1);
%   fullPath = fullfile('data/','tortoiseROCs2caffe',s1);
    fullPath = fullfile('data/','tortoiseROCs3verydeep',s1);
    savefig(fullPath);
    T{z} = info2.auc;
    Tpr{z} = tpr;
    Fpr{z} = fpr;
    Score{z}= testScores;
end
