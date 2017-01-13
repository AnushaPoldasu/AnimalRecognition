load('data/dogROCs3verydeep/rotate/testLabels.mat');
run vlfeat-0.9.20/toolbox/vl_setup;
load('data/dogROCs3verydeep/rotate/DScore.mat');

A1 = [DScore{1}];
[tpr1,fpr1,info1] = vl_roc(testLabels,A1,'plot','fptp');
auc1 = sprintf('%3.3f',info1.auc);

A2 = [DScore{2}];
[tpr2,fpr2,info2] = vl_roc(testLabels,A2,'plot','fptp');
auc2 = sprintf('%3.3f',info2.auc);
figure;
plot(fpr1,tpr1,'-k.', 'LineWidth', 5);
hold on
plot(fpr2,tpr2,'-m.', 'LineWidth', 5);
title('ROC curve for Dogs dataset')
set(gca,'FontSize',20);
xlabel('False Positive Rate', 'FontSize', 20)
ylabel('True Positive Rate','FontSize', 20)
legend('before data augmentation','after data augmentation');
legend(['before data augmentation, auc1=',auc1],['after data augmentation, auc2=',auc2]);