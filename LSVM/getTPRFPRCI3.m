function getTPRFPRCI3
setup;
load('data/dogROCsO/test/Score.mat'); %change to the path of saved score mat file
D1 = DScore;
load('data/moutaingoatROCs/Score.mat')
D2 = MScore;
load('data/tortoiseROCs/Score.mat')
D3 = TScore;

for i = 1: 50
    testScores1 = [D1{i}];
    testScores2 = [D2{i}];
    testScores3 = [D3{i}];
    % change all values to 0 - 1
    for j = 1: 200
        if testScores1(j) < 0
            testScores1(j)= testScores1(j)+1;
        end
        if testScores2(i,j) < 0
            testScores2(i,j) = testScores2(i,j) +1;
        end
        if testScores3(j) < 0
            testScores3(j)= testScores3(j)+1;
        end
    end
    p1 = testScores1(1:100);
    n1 = testScores1(101:200);
    p2 = testScores2(i, 1:100);
    n2 = testScores2(i, 101:200);
    p3 = testScores3(1:100);
    n3 = testScores3(101:200);
    tpr1 = zeros(1,20);
    fpr1 = zeros(1,20);
    tpr2 = zeros(1,20);
    fpr2 = zeros(1,20);
    tpr3 = zeros(1,20);
    fpr3 = zeros(1,20);
    % calculate positive
    for t = 1:20
        tp=p1(find(p1>(t/20)));
        fp=n1(find(n1>(t/20)));
        tpr1(t) = size(tp,2)/size(p1,2);
        fpr1(t) = size(fp,2)/size(n1,2);
        tp=p2(find(p2>(t/20)));
        fp=n2(find(n2>(t/20)));
        tpr2(t) = size(tp,2)/size(p2,2);
        fpr2(t) = size(fp,2)/size(n2,2);
        tp=p3(find(p3>(t/20)));
        fp=n3(find(n3>(t/20)));
        tpr3(t) = size(tp,2)/size(p3,2);
        fpr3(t) = size(fp,2)/size(n3,2);
    end
    if i == 1
       TPRA1 = tpr1;
       FPRA1 = fpr1;
       TPRA2 = tpr2;
       FPRA2 = fpr2;
       TPRA3 = tpr3;
       FPRA3 = fpr3;
    else
       TPRA1 = cat(1,TPRA1,tpr1);
       FPRA1 = cat(1,FPRA1,fpr1);
       TPRA2 = cat(1,TPRA2,tpr2);
       FPRA2 = cat(1,FPRA2,fpr2);
       TPRA3 = cat(1,TPRA3,tpr3);
       FPRA3 = cat(1,FPRA3,fpr3);
    end
end
tprCi1 = zeros(2,20);
tprCi2 = zeros(2,20);
tprCi3 = zeros(2,20);
meanTPR1 = mean(TPRA1);
meanTPR2 = mean(TPRA2);
meanTPR3 = mean(TPRA3);
fprCi1 = zeros(2,20);
fprCi2 = zeros(2,20);
fprCi3 = zeros(2,20);
meanFPR1 = mean(FPRA1);
meanFPR2 = mean(FPRA2);
meanFPR3 = mean(FPRA3);
for i = 1: 20
    tprCi1(:,i)= CI(TPRA1(:,i));
    fprCi1(:,i)= CI(FPRA1(:,i));
    tprCi2(:,i)= CI(TPRA2(:,i));
    fprCi2(:,i)= CI(FPRA2(:,i));
    tprCi3(:,i)= CI(TPRA3(:,i));
    fprCi3(:,i)= CI(FPRA3(:,i));
end
threshold = 1:20;
threshold = threshold/20;
figure
pos1 =tprCi1(2,:) - meanTPR1;
neg1 =tprCi1(1,:) - meanTPR1;
pos2 =tprCi2(2,:) - meanTPR2;
neg2 =tprCi2(1,:) - meanTPR2;
pos3 =tprCi3(2,:) - meanTPR3;
neg3 =tprCi3(1,:) - meanTPR3;
plot(threshold, meanTPR1, '-k.', 'LineWidth', 5)
hold on
plot(threshold, meanTPR2, '-m.', 'LineWidth', 5)
hold on
plot(threshold, meanTPR3, '-c.', 'LineWidth', 5)
hold on
errorbar(threshold,meanTPR1,neg1,pos1,'o','LineWidth', 3)
hold on
errorbar(threshold,meanTPR2,neg2,pos2,'o','LineWidth', 3)
hold on
errorbar(threshold,meanTPR3,neg3,pos3,'o','LineWidth', 3)
xlim([0 1.1])
ylim([0 1.1])
title('Dogs dataset')
% title('MountainGoat dataset')
% title('Tortoise dataset')
set(gca,'FontSize',20);
xlabel('threshold', 'FontSize', 40)
ylabel('True Positive Rate','FontSize', 40)
legend('dogs','moutain goat', 'tortoise');

figure
pos1 =fprCi1(2,:) - meanFPR1;
neg1 =fprCi1(1,:) - meanFPR1;
pos2 =fprCi2(2,:) - meanFPR2;
neg2 =fprCi2(1,:) - meanFPR2;
pos3 =fprCi3(2,:) - meanFPR3;
neg3 =fprCi3(1,:) - meanFPR3;
plot(threshold, meanFPR1, '-k.', 'LineWidth', 5)
hold on
plot(threshold, meanFPR2, '-m.', 'LineWidth', 5)
hold on
plot(threshold, meanFPR3, '-c.', 'LineWidth', 5)
hold on
errorbar(threshold,meanFPR1,neg1,pos1,'o','LineWidth', 3)
hold on
errorbar(threshold,meanFPR2,neg2,pos2,'o','LineWidth', 3)
hold on
errorbar(threshold,meanFPR3,neg3,pos3,'o','LineWidth', 3)
xlim([0 1.1])
ylim([0 1.1])
title('Dogs dataset')
% title('MountainGoat dataset')
% title('Tortoise dataset')
set(gca,'FontSize',20);
xlabel('threshold', 'FontSize', 40)
ylabel('False Positive Rate','FontSize', 40)
legend('dogs','moutain goat', 'tortoise');

function A = CI(x)
SEM = std(x)/sqrt(length(x));               % Standard Error
ts = tinv([0.025  0.975],length(x)-1);      % T-Score
A = mean(x) + ts*SEM;   
 
 