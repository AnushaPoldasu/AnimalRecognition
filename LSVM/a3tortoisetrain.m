
% --------------------------------------------------------------------
% train the tortoise dataset
% --------------------------------------------------------------------
setup;
T = cell(1,75); %store the testing ROC AUC value for 75 times
Tpr = cell(1,75);
Tnr = cell(1,75);
Score = cell(1,75); %store the testing score or each images for 75 times
for z = 1:75
    % --------------------------------------------------------------------
    % Stage I: Data Preparation
    % --------------------------------------------------------------------
    % step 1 get the splitted training and testing images
    fprintf('%s\n','slpit files');
    % positive data
    spath = 'data/tortoise_all';
    dpath1 = 'data/tortoiseTrainI';
    dpath2 = 'data/tortoiseTestI';
    randomSplitImgFile(spath, dpath1,dpath2,100); % spath copy to dpath1, dpath1 move 100 to dpath2
    
    % negative data
    spath = 'data/n_all';
    dpath1 = 'data/nTrainIt';
    dpath2 = 'data/nTestIt';
    randomSplitImgFile(spath, dpath1,dpath2,100);
    %----------------------------------------------------------------------
    % step 2 generate histograms for tortoise and negative samples
    % tortoisetest
    path = 'data/tortoiseTestI';
    tortoisetest_names = getImageSet(path);
    vocabulary = computeVocabularyFromImageList(tortoisetest_names);
    tortoise_test_hist = computeHistogramsFromImageList(vocabulary, tortoisetest_names);
    % tortoisetrain
    path = 'data/tortoiseTrainI';
    tortoisetrain_names = getImageSet(path);
    vocabulary = computeVocabularyFromImageList(tortoisetrain_names);
    tortoise_train_hist = computeHistogramsFromImageList(vocabulary, tortoisetrain_names);
    % ntest
    path = 'data/nTestIt';
    ntestt_names = getImageSet(path);
    vocabulary = computeVocabularyFromImageList(ntestt_names);
    n_testt_hist = computeHistogramsFromImageList(vocabulary, ntestt_names);
    % ntrain
    path = 'data/nTrainIt';
    ntraint_names = getImageSet(path);
    vocabulary = computeVocabularyFromImageList(ntraint_names);
    n_traint_hist = computeHistogramsFromImageList(vocabulary, ntraint_names);
    %----------------------------------------------------------------------
    % step3 save the histograms and names
    filename5='data/tortoise_train_hist';
    save(filename5,'tortoise_train_hist','tortoisetrain_names');
    filename6='data/tortoise_test_hist';
    save(filename6,'tortoise_test_hist','tortoisetest_names');
    filename7='data/n_traint_hist';
    save(filename7,'n_traint_hist','ntraint_names');
    filename8='data/n_testt_hist';
    save(filename8,'n_testt_hist','ntestt_names');
    %----------------------------------------------------------------------
    % step4
    % Load training data
    pos = load('data/tortoise_train_hist.mat') ;
    neg = load('data/n_traint_hist.mat') ;
    names = {pos.tortoisetrain_names{:}, neg.ntraint_names{:}};
    histograms = [pos.tortoise_train_hist, neg.n_traint_hist] ;
    labels = [ones(1,numel(pos.tortoisetrain_names)), - ones(1,numel(neg.ntraint_names))] ;
    clear pos neg;

    % Load testing data
    pos = load('data/tortoise_test_hist.mat') ;
    neg = load('data/n_testt_hist.mat') ;
    testNames = {pos.tortoisetest_names{:}, neg.ntestt_names{:}};
    testHistograms = [pos.tortoise_test_hist, neg.n_testt_hist] ;
    testLabels = [ones(1,numel(pos.tortoisetest_names)), - ones(1,numel(neg.ntestt_names))] ;
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
    s = 'rocTortoise';
    s1= strcat(s,sprintf('%d%s',z,'.fig'));
    fullPath = fullfile('data','tortoiseROCs',s1) 
    savefig(fullPath);
    T{z} = info2.auc;
    Tpr{z} = tpr;
    Tnr{z} = tnr;
    Score{z}= testScores;
end