function [tprCi, fprCi] = getTPRFPRCI
load('data/dogROCs1vgg/DScore.mat')
D1 = DScore;
load('data/dogROCs2caffe/DScore.mat')
D2 = DScore;
load('data/dogROCs3veryDeep/DScore.mat')
D3 = DScore;
for i = 1: 50
    testScores = [D3{i}];
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
tprCi = zeros(2,20);
meanTPR = mean(TPRA);
fprCi = zeros(2,20);
meanFPR = mean(FPRA);
for i = 1: 20
    tprCi(:,i)= CI(TPRA(:,i));
    fprCi(:,i)= CI(FPRA(:,i));
end
threshold = 1:20;
threshold = threshold/20;
figure
pos =tprCi(2,:) - meanTPR;
neg =tprCi(1,:) - meanTPR;
plot(threshold, meanTPR, '-k.', 'LineWidth', 5)
hold on
errorbar(threshold,meanTPR,neg,pos,'o','LineWidth', 3)
xlim([0 1.1])
ylim([0 1])
title('Moutaingoat dataset')
set(gca,'FontSize',30);
xlabel('threshold', 'FontSize', 40)
ylabel('True Positive Rate','FontSize', 40)

figure
pos =fprCi(2,:) - meanFPR;
neg =fprCi(1,:) - meanFPR;
plot(threshold, meanFPR, '-k.', 'LineWidth', 5)
hold on
errorbar(threshold,meanFPR,neg,pos,'o','LineWidth', 3)
xlim([0 1.1])
ylim([0 1])
title('Moutaingoat dataset')
set(gca,'FontSize',30);
xlabel('threshold', 'FontSize', 40)
ylabel('False Positive Rate','FontSize', 40)
function A = CI(x)
SEM = std(x)/sqrt(length(x));               % Standard Error
ts = tinv([0.025  0.975],length(x)-1);      % T-Score
A = mean(x) + ts*SEM;   
 
    