% --------------------------------------------------------------------
% train the mountain goat dataset
% --------------------------------------------------------------------
setup ;
M = cell(1,75); %store the testing ROC AUC value for 75 times
MTpr = cell(1,75);
MTnr = cell(1,75);
MScore = cell(1,75); %store the testing score or each images for 75 times

for y = 1:75
    % --------------------------------------------------------------------
    % Stage I: Data Preparation
    % --------------------------------------------------------------------
    % step 1 get the splitted training and testing images
    fprintf('%s\n','slpit files');
    % positive data
    spath = 'data/moutaingoat_all';
    dpath1 = 'data/moutaingoatTrainI';
    dpath2 = 'data/moutaingoatTestI';
    randomSplitImgFile(spath, dpath1,dpath2,100); % spath copy to dpath1, dpath1 move 100 to dpath2

    % negative data
    spath = 'data/n_all';
    dpath1 = 'data/nTrainI';
    dpath2 = 'data/nTestI';
    randomSplitImgFile(spath, dpath1,dpath2,100);
    %----------------------------------------------------------------------
    % step 2 generate histograms for mountaingoat and negative samples
    % mountaingoattest
    path = 'data/moutaingoatTestI';
    mougoattest_names = getImageSet(path);
    vocabulary = computeVocabularyFromImageList(mougoattest_names);
    mougoat_test_hist = computeHistogramsFromImageList(vocabulary, mougoattest_names);
    % mountaingoattrain
    path = 'data/moutaingoatTrainI';
    mougoattrain_names = getImageSet(path);
    vocabulary = computeVocabularyFromImageList(mougoattrain_names);
    mougoat_train_hist = computeHistogramsFromImageList(vocabulary, mougoattrain_names);
    % ntest
    path = 'data/nTestI';
    ntest_names = getImageSet(path);
    vocabulary = computeVocabularyFromImageList(ntest_names);
    n_test_hist = computeHistogramsFromImageList(vocabulary, ntest_names);
    % ntrain
    path = 'data/nTrainI';
    ntrain_names = getImageSet(path);
    vocabulary = computeVocabularyFromImageList(ntrain_names);
    n_train_hist = computeHistogramsFromImageList(vocabulary, ntrain_names);
    %----------------------------------------------------------------------
    % step 3 save the histograms and names
    filename3='data/mougoat_train_hist';
    save(filename3,'mougoat_train_hist','mougoattrain_names');
    filename4='data/mougoat_test_hist';
    save(filename4,'mougoat_test_hist','mougoattest_names');
    filename7='data/n_train_hist';
    save(filename7,'n_train_hist','ntrain_names');
    filename8='data/n_test_hist';
    save(filename8,'n_test_hist','ntest_names');
    %----------------------------------------------------------------------
    % step 4
    % Load training data
    pos = load('data/mougoat_train_hist.mat');
    neg = load('data/n_train_hist.mat');
    names = {pos.mougoattrain_names{:}, neg.ntrain_names{:}};
    histograms = [pos.mougoat_train_hist, neg.n_train_hist] ;
    labels = [ones(1,numel(pos.mougoattrain_names)), - ones(1,numel(neg.ntrain_names))];
    clear pos neg;

    % Load testing data
    pos = load('data/mougoat_test_hist.mat');
    neg = load('data/n_test_hist.mat') ;
    testNames = {pos.mougoattest_names{:}, neg.ntest_names{:}};
    testHistograms = [pos.mougoat_test_hist, neg.n_test_hist] ;
    testLabels = [ones(1,numel(pos.mougoattest_names)), - ones(1,numel(neg.ntest_names))];
    clear pos neg;

    % throw away part of the training data
    % fraction = .1;
    % fraction = .5;
    fraction = +inf;

    % append labels to historgrams
    sel = vl_colsubset(1:numel(labels), fraction, 'uniform');
    names = names(sel);
    histograms = histograms(:,sel);
    labels = labels(:,sel);
    clear sel;

    % count how many images are there
    fprintf('Number of training images: %d positive, %d negative\n', sum(labels > 0), sum(labels < 0));
    fprintf('Number of testing images: %d positive, %d negative\n', sum(testLabels > 0), sum(testLabels < 0));

    % normalize the histograms before running the linear SVM
    histograms = bsxfun(@times, histograms, 1./sqrt(sum(histograms.^2,1)));
    testHistograms = bsxfun(@times, testHistograms, 1./sqrt(sum(testHistograms.^2,1)));

    % --------------------------------------------------------------------
    % Stage II: Training a classifier 
    % --------------------------------------------------------------------
    C = 10;
    [w, bias] = trainLinearSVM(histograms, labels, C) ;
    % --------------------------------------------------------------------
    % Stage III: Classify the test images and assess the performance
    % --------------------------------------------------------------------
    % Test the linar SVM
    testScores = w' * testHistograms + bias;
    % --------------------------------------------------------------------
    % Visualize the roc curve
    figure(1) ; clf ; set(1,'name','Receiver Operating Characteristic on test data');
    vl_roc(testLabels, testScores);
    % Print results
    % --------------------------------------------------------------------
    [tpr, fpr, info2] = vl_roc(testLabels, testScores);
    fprintf('Test ROC: %.3f\n', info2.auc);
    s = 'rocMoutainGoat';
    s1= strcat(s,sprintf('%d%s',y,'.fig'));
    fullPath = fullfile('data','moutaingoatROCs',s1);
    savefig(fullPath);
    M{y} = info2.auc;
    MTpr{y} = tpr;
    MTnr{y} = tnr;
    MScore{y}= testScores;
end
mscore='data/moutaingoatROCs/MScore';
save(mscore,'MScore');