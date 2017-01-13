% --------------------------------------------------------------------
% train the dog dataset
% --------------------------------------------------------------------
setup;
D = cell(1,75);    %store the testing ROC AUC value for 75 times
DTpr = cell(1,75);
DFpr = cell(1,75);
DScore = cell(1,75); %store the testing score or each images for 75 times

for x = 1: 1
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
    % step 2 generate histograms for dogs and negative samples
    % dogtest
    fprintf('%s\n','generate histograms');
    path = 'data/dogsTestI';
    % path = 'data/cpImagePos';
    dogtest_names = getImageSet(path);
    vocabulary = computeVocabularyFromImageList(dogtest_names);
    dogs_test_hist = computeHistogramsFromImageList(vocabulary, dogtest_names);
    % dogtrain
    path = 'data/dogsTrainI';
    dogtrain_names = getImageSet(path);
    vocabulary = computeVocabularyFromImageList(dogtrain_names);
    dogs_train_hist = computeHistogramsFromImageList(vocabulary, dogtrain_names);
    % ntest
    path = 'data/nTestId';
    % path = '/Users/junzhi/Documents/MATLAB/Classification/LSVM/data/cpImageNeg';
    ntestd_names = getImageSet(path);
    vocabulary = computeVocabularyFromImageList(ntestd_names);
    n_testd_hist = computeHistogramsFromImageList(vocabulary, ntestd_names);
    % ntrain
    path = 'data/nTrainId';
    ntraind_names = getImageSet(path);
    vocabulary = computeVocabularyFromImageList(ntraind_names);
    n_traind_hist = computeHistogramsFromImageList(vocabulary, ntraind_names);
    %----------------------------------------------------------------------
    % step 3 save the histograms and names
    fprintf('%s\n','save the histograms files');
    filename1='data/dogs_train_hist';
    save(filename1,'dogs_train_hist','dogtrain_names');
    filename2='data/dogs_test_hist';
    save(filename2,'dogs_test_hist','dogtest_names');
    filename7='data/n_traind_hist';
    save(filename7,'n_traind_hist','ntraind_names');
    filename8='data/n_testd_hist';
    save(filename8,'n_testd_hist','ntestd_names');
    %----------------------------------------------------------------------
    % step 4
    %Load training data
    pos = load('data/dogs_train_hist.mat') ;
    neg = load('data/n_traind_hist.mat') ;
    names = {pos.dogtrain_names{:}, neg.ntraind_names{:}};
    histograms = [pos.dogs_train_hist, neg.n_traind_hist] ;
    labels = [ones(1,numel(pos.dogtrain_names)), - ones(1,numel(neg.ntraind_names))] ;
    clear pos neg;

    % Load testing data
    pos = load('data/dogs_test_hist.mat') ;
    neg = load('data/n_testd_hist.mat') ;
    testNames = {pos.dogtest_names{:}, neg.ntestd_names{:}};
    testHistograms = [pos.dogs_test_hist, neg.n_testd_hist] ;
    testLabels = [ones(1,numel(pos.dogtest_names)), - ones(1,numel(neg.ntestd_names))] ;
    clear pos neg;

    % throw away part of the training data
    % fraction = .1 ;
    % fraction = .5 ;
    fraction = +inf ;

    % append labels to historgrams
    sel = vl_colsubset(1:numel(labels), fraction, 'uniform') ;
    names = names(sel) ;
    histograms = histograms(:,sel) ;
    labels = labels(:,sel) ;
    clear sel ;

    % count how many images are there
    fprintf('Number of training images: %d positive, %d negative\n', sum(labels > 0), sum(labels < 0)) ;
    fprintf('Number of testing images: %d positive, %d negative\n', sum(testLabels > 0), sum(testLabels < 0)) ;

    % normalize the histograms before running the linear SVM
    histograms = bsxfun(@times, histograms, 1./sqrt(sum(histograms.^2,1))) ;
    testHistograms = bsxfun(@times, testHistograms, 1./sqrt(sum(testHistograms.^2,1))) ;

    % --------------------------------------------------------------------
    % Stage II: Training a classifier 
    % --------------------------------------------------------------------
    C = 10 ;
    [w, bias] = trainLinearSVM(histograms, labels, C) ;
    % --------------------------------------------------------------------
    % Stage III: Classify the test images and assess the performance
    % --------------------------------------------------------------------
    % Test the linar SVM
    testScores = w' * testHistograms + bias ;
    % --------------------------------------------------------------------
    % Visualize the roc curve
    figure(1) ; clf ; set(1,'name','Receiver Operating Characteristic on test data') ;
    vl_roc(testLabels, testScores) ;
    % Print results
    % --------------------------------------------------------------------
    [tpr, fpr, info2] = vl_roc(testLabels, testScores) ;
    fprintf('Test ROC: %.3f\n', info2.auc) ;
    s = 'rocDog';
    s1= strcat(s,sprintf('%d%s',x,'.fig'));
    fullPath = fullfile('data/','dogROCs',s1) 
    savefig(fullPath);
    D{x} = info2.auc;
    DTpr{x} = tpr;
    DFpr{x} = fpr;
    DScore{x}= testScores;
end
dscore='data/dogROCs/DScore';

save(dscore,'DScore');

