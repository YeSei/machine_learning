function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

values = [0.01 0.03 0.1 0.3 1 3 10 30];
min_err = 1;
for i=1:8,
    for j=1:8,
        C_test = values(i);   %C参数的预测值
        sigma_test = values(j);    %sigma参数的预测值
        model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));  %把C参数和训练样本通过svmTrain训练,把sigma参数和训练样本通过gaussianKernel训练
        predictions = svmPredict(model, Xval);   %通过svmPredict得到预测值
        err = mean(double(predictions ~= yval)); %对比预测值和真实值得到误差
        if err < min_err,            %假如误差小于最小误差
            C = C_test;              %C = C参数的预测值
            sigma = sigma_test;      %sigma = sigma参数的预测值
            min_err = err;           %最小误差 = 前一次误差
        end
    end
end


% =========================================================================

end
