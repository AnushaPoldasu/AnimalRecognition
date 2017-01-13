function [w,bias] = trainLinearSVM(x,y,C)
% learns an SVM from data vectors X [D (examples) * N (features)] and
% labels [N]  =>  w, bias that W'*X(:,i)+B has the same sign of LABELS(i)

lambda = 1 / (C * numel(y)) ;
[w, bias] = vl_svmtrain(single(x),y, lambda) ;
