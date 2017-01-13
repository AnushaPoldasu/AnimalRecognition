% plot ROC curves for three animals (choose index of typical testing scores)
load('data/dogROCsO/test/Score.mat'); %change to the path of saved score mat file
D1 = DScore;
load('data/moutaingoatROCs/Score.mat')
D2 = MScore;
load('data/tortoiseROCs/Score.mat')
D3 = TScore;

setup;
testLabels =[ones(1,numel(D1{1})/2), - ones(1,numel(D1{1})/2)] ;
A1 = [D1{7}];
[tpr1,fpr1,info1] = vl_roc(testLabels,A1);
auc1 = sprintf('%3.4f',info1.auc);

A2 = D2(14,1:200);
[tpr2,fpr2,info2] = vl_roc(testLabels,A2);
auc2 = sprintf('%3.4f',info2.auc);

A3 = [D3{1}];
[tpr3,fpr3,info3] = vl_roc(testLabels,A3);
auc3 = sprintf('%3.4f',info3.auc);

figure;
plot(fpr1,tpr1,'-k.', 'LineWidth', 5);
hold on
plot(fpr2,tpr2,'-m.', 'LineWidth', 5);
hold on
plot(fpr3,tpr3,'-c.', 'LineWidth', 5);

title('ROC curve for classification using LSVM with dense SIFT features')

set(gca,'FontSize',20);
xlabel('False Positive Rate', 'FontSize', 20)
ylabel('True Positive Rate','FontSize', 20)
legend('dogs','moutain goat', 'tortoise');
legend(['auc of dogRoc: ',auc1],['auc of moutaingoatROCs: ',auc2], ['auc of tortoiseROCs ',auc3]);