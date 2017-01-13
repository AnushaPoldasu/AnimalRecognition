load('/Users/junzhi/Documents/MATLAB/classifier1/classfier1-LSVM/data/tortoiseROCs/testScores.mat')
testLabels = [ones(1,100), - ones(1,100)] ;
for i = 1: 50
    testScores = [Score{i}];
    % change all values to 0 - 1
    for j = 1: 200
        if testScores(j) < 0
            testScores(j)= testScores(j)+1;
        end
    end
    p = testScores(1:100);
    n = testScores(101:200);
    tpr = zeros(1,20);
    fpr = zeros(1,20);
    % calculate positive
    for t = 1:20
        tp=p(find(p>(t/20)));
        fp=n(find(n>(t/20)));
        tpr(t) = size(tp,2)/size(p,2);
        fpr(t) = size(fp,2)/size(n,2);
    end
    if i == 1
       TPRA = tpr;
       FPRA = fpr;
    else
       TPRA = cat(1,TPRA,tpr);
       FPRA = cat(1,FPRA,fpr);
    end
end
    