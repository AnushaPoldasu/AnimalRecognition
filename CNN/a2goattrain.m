% --------------------------------------------------------------------
% train the mountain goat dataset
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
encoding = 'vggv16-fc7';
encoder = loadEncoderVeryDeep(encoding);

M = cell(1,100);
MTpr = cell(1,100);
MFpr = cell(1,100);
MScore = cell(1,100);
for y = 1:100
    % --------------------------------------------------------------------
    % Stage I: Data Preparation
    % --------------------------------------------------------------------
%     % step 1 get the splitted training and testing images
%     fprintf('%s\n','slpit files');
%     % positive data
%     spath = 'data/moutaingoat_all';
%     dpath1 = 'data/moutaingoatTrainI';
%     dpath2 = 'data/moutaingoatTestI';
%     randomSplitImgFile(spath, dpath1,dpath2,100); % spath copy to dpath1, dpath1 move 100 to dpath2
% 
%     % negative data
%     spath = 'data/n_all';
%     dpath1 = 'data/nTrainI';
%     dpath2 = 'data/nTestI';
%     randomSplitImgFile(spath, dpath1,dpath2,100);
%     %----------------------------------------------------------------------
%     % step 2 generate descriptors for mountaingoat and negative samples
%     % mountaingoattest
%     fprintf('%s\n','generate descriptors');
%     path = 'data/moutaingoatTestI';
%     mougoattest_names = getImageSet(path);
%     mougoattest_descriptors = computepsisFromImageList(encoder, mougoattest_names);
%     % mountaingoattrain
%     path = 'data/moutaingoatTrainI';
%     mougoattrain_names = getImageSet(path);
%     mougoattrain_descriptors = computepsisFromImageList(encoder, mougoattrain_names);
%     % ntest
%     path = 'data/nTestI';
%     ntestM_names = getImageSet(path);
%     ntestM_descriptors = computepsisFromImageList(encoder, ntestM_names);
%     % ntrain
%     path = 'data/nTrainI';
%     ntrainM_names = getImageSet(path);
%     ntrainM_descriptors = computepsisFromImageList(encoder,ntrainM_names);
%     %----------------------------------------------------------------------
%     % step 3  save the descriptors and names
%     filename3='data/mougoat_train_psis';
%     save(filename3,'mougoattrain_descriptors','mougoattrain_names');
%     filename4='data/mougoat_test_psis';
%     save(filename4,'mougoattest_descriptors','mougoattest_names');
%     filename7='data/nM_train_psis';
%     save(filename7,'ntrainM_descriptors','ntrainM_names');
%     filename8='data/nM_test_psis';
%     save(filename8,'ntestM_descriptors','ntestM_names');
    %----------------------------------------------------------------------
    % step 4
    % Load training data    
    numPos = +inf;
    numNeg = +inf;
    names = {};
    descriptors = [];
    labels = [];
    pos = load('data/mougoat_train_psis.mat');
    neg = load('data/nM_train_psis.mat');
    selp = vl_colsubset(1:numel(pos.mougoattrain_names),numPos,'beginning');
    seln = vl_colsubset(1:numel(neg.ntrainM_names),numNeg,'beginning');
    names = horzcat(names, pos.mougoattrain_names(selp), neg.ntrainM_names(seln));
    descriptors = horzcat(descriptors, pos.mougoattrain_descriptors(:,selp), neg.ntrainM_descriptors(:,seln));
    labels = horzcat(labels, ones(1,numel(selp)), - ones(1,numel(seln)));
    clear pos neg;
    
    % Load testing data
    testNames = {};
    testDescriptors = [];
    testLabels = [];
    pos = load('data/mougoat_test_psis.mat');
    neg = load('data/nM_test_psis.mat');
    selp = vl_colsubset(1:numel(pos.mougoattest_names),numPos,'beginning');
    seln = vl_colsubset(1:numel(neg.ntestM_names),numNeg,'beginning');
    testNames = {pos.mougoattest_names{:}, neg.ntestM_names{:}};
    testLabels = [ones(1,numel(pos.mougoattest_names)), - ones(1,numel(neg.ntestM_names))];
    testDescriptors = cat(3, testDescriptors, [pos.mougoattest_descriptors, neg.ntestM_descriptors]);
    clear pos neg;

    % count how many images are there
    fprintf('Number of training images: %d positive, %d negative\n', sum(labels > 0), sum(labels < 0)) ;
    fprintf('Number of testing images: %d positive, %d negative\n', sum(testLabels > 0), sum(testLabels < 0)) ;

    % normalize the histograms before running the linear SVM
    descriptors = bsxfun(@times, descriptors, 1./sqrt(sum(descriptors.^2,1))) ;
    testDescriptors = bsxfun(@times, testDescriptors, 1./sqrt(sum(testDescriptors.^2,1))) ;

    % --------------------------------------------------------------------
    % Stage II: Training a classifier 
    % --------------------------------------------------------------------
    C = 10 ;
    [w, bias] = trainLinearSVM(descriptors, labels, C) ;
    % --------------------------------------------------------------------
    % Stage III: Classify the test images and assess the performance
    % --------------------------------------------------------------------
    % Test the linar SVM
    testScores = w' * testDescriptors + bias ;
    % --------------------------------------------------------------------
    % Visualize the roc curve
    figure(1) ; clf ; set(1,'name','Receiver Operating Characteristic on test data') ;
    vl_roc(testLabels, testScores) ;
    % Print results
    % --------------------------------------------------------------------
    [tpr, tnr, info2] = vl_roc(testLabels, testScores) ;
    fprintf('Test ROC: %.3f\n', info2.auc) ;
    s = 'rocMoutainGoat';
    s1= strcat(s,sprintf('%d%s',y,'.fig'));
%   fullPath = fullfile('data/','moutaingoatROCs1vgg',s1);
%   fullPath = fullfile('data/','moutaingoatROCs2caffe',s1);
    fullPath = fullfile('data/','moutaingoatROCs3verydeep',s1);
    savefig(fullPath);
    M{y} = info2.auc;
    MTpr{y} = tpr;
    MFpr{y} = tnr;
    MScore{y}= testScores;
end
