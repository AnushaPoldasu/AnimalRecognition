
load('data/tortoiseROCs1vgg/Score.mat')
D1 = Score; %89
load('data/tortoiseROCs2caffe/Score.mat')
D2 = Score; %25
load('data/tortoiseROCs3veryDeep/Score.mat')
D3 = Score; %55

setup;
testLabels =[ones(1,numel(D1{1})/2), - ones(1,numel(D1{1})/2)] ;
A1 = [D1{89}];
[tpr1,fpr1,info1] = vl_roc(testLabels,A1);
auc1 = sprintf('%3.4f',info1.auc);

A2 = [D2{25}];
[tpr2,fpr2,info2] = vl_roc(testLabels,A2);
auc2 = sprintf('%3.4f',info2.auc);

A3 = [D3{55}];
[tpr3,fpr3,info3] = vl_roc(testLabels,A3);
auc3 = sprintf('%3.4f',info3.auc);

figure;
plot(fpr1,tpr1,'-k.', 'LineWidth', 5);
hold on
plot(fpr2,tpr2,'-m.', 'LineWidth', 5);
hold on
plot(fpr3,tpr3,'-c.', 'LineWidth', 5);

% title('ROC curve for Dogs dataset')
title('ROC curve for Mountain goat dataset')
% title('ROC curve for Tortoise dataset')

set(gca,'FontSize',20);
xlabel('False Positive Rate', 'FontSize', 20)
ylabel('True Positive Rate','FontSize', 20)
legend('VGG-128','caffe','VGG-verydeep-16');
legend(['auc using encoder VGG-128: ',auc1],['auc using encoder caffe: ',auc2], ['VGG-verydeep-16: ',auc3]);