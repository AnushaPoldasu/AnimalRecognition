% --------------------------------------------------------------------
% train the dog dataset
% --------------------------------------------------------------------
setup;
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
encoding = 'vggv16-fc7';
encoder = loadEncoderVeryDeep(encoding);

D = cell(1,1);
DTpr = cell(1,100);
DFpr = cell(1,100);
DScore = cell(1,100);
for x = 1: 100
    % --------------------------------------------------------------------
    % Stage I: Data Preparation
    % --------------------------------------------------------------------
    % step 1 get the splitted training and testing images
    fprintf('%s\n','slpit files');
    % positive data
    spath = 'data/dogs_all';
    dpath1 = 'data/dogsTrainI';
    dpath2 = 'data/dogsTestI';
    randomSplitImgFile(spath, dpath1,dpath2,100); % spath copy to dpath1, dpath1 move 100 to dpath2

    % negative data
    spath = 'data/n_all';
    dpath1 = 'data/nTrainId';
    dpath2 = 'data/nTestId';
    randomSplitImgFile(spath, dpath1,dpath2,100);
    %----------------------------------------------------------------------
    % step 2 generate descriptors for dogs and negative samples
    % dogtest
    fprintf('%s\n','generate descriptors');
    path = 'data/dogsTestI';
    dogtest_names = getImageSet(path);
    dogtest_descriptors = computepsisFromImageList(encoder, dogtest_names);
    % dogtrain
    path = 'data/dogsTrainI';
    dogtrain_names = getImageSet(path);
    dogtrain_descriptors = computepsisFromImageList(encoder, dogtrain_names);
    % ntest
    path = 'data/nTestId';
    ntestd_names = getImageSet(path);
    ntest_descriptors = computepsisFromImageList(encoder, ntestd_names);
    % ntrain
    path = 'data/nTrainId';
    ntraind_names = getImageSet(path);
    ntrain_descriptors = computepsisFromImageList(encoder, ntraind_names);
    %----------------------------------------------------------------------
    % step 3 save the descriptors and names
    fprintf('%s\n','save the psi files');
    filename1='data/dogs_train_psis';
    save(filename1,'dogtrain_descriptors','dogtrain_names');
    filename2='data/dogs_test_psis';
    save(filename2,'dogtest_descriptors','dogtest_names');
    filename7='data/n_traind_psis';
    save(filename7,'ntrain_descriptors','ntraind_names');
    filename8='data/n_testd_psis';
    save(filename8,'ntest_descriptors','ntestd_names');
    %----------------------------------------------------------------------
    % step 4
    % Load training data
    numPos = +inf ;
    numNeg = +inf ;
    names = {} ;
    descriptors = [] ;
    labels = [] ;
    pos = load('data/dogs_train_psis.mat');
    neg = load('data/n_traind_psis.mat');

    selp = vl_colsubset(1:numel(pos.dogtrain_names),numPos,'beginning');
    seln = vl_colsubset(1:numel(neg.ntraind_names),numNeg,'beginning');
    names = horzcat(names, pos.dogtrain_names(selp), neg.ntraind_names(seln));
    descriptors = horzcat(descriptors, pos.dogtrain_descriptors(:,selp), neg.ntrain_descriptors(:,seln));
    labels = horzcat(labels, ones(1,numel(selp)), - ones(1,numel(seln)));
    clear pos neg;

    % Load testing data
    testNames = {};
    testDescriptors = [];
    testLabels = [];
    pos = load('data/dogs_test_psis.mat');
    neg = load('data/n_testd_psis.mat');
    selp = vl_colsubset(1:numel(pos.dogtest_names),numPos,'beginning');
    seln = vl_colsubset(1:numel(neg.ntestd_names),numNeg,'beginning');
    testNames = {pos.dogtest_names{:}, neg.ntestd_names{:}};
    testLabels = [ones(1,numel(pos.dogtest_names)), - ones(1,numel(neg.ntestd_names))];
    testDescriptors = cat(3, testDescriptors, [pos.dogtest_descriptors, neg.ntest_descriptors]);
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
    C = 10;
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
    [tpr, fpr, info2] = vl_roc(testLabels, testScores);
    fprintf('Test ROC: %.3f\n', info2.auc);
    s = 'rocDog';
    s1= strcat(s,sprintf('%d%s',x,'.fig'));
%   fullPath = fullfile('data/','dogROCs1vgg',s1);
%   fullPath = fullfile('data/','dogROCs2caffe',s1);
    fullPath = fullfile('data/','dogROCs3verydeep',s1);
    savefig(fullPath);
    D{x} = info2.auc;
    DTpr{x} = tpr;
    DFpr{x} = fpr;
    DScore{x}= testScores;
end

