function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y
%   J存储的值作为函数返回值。

% Initialize some useful values
m = length(y);  % number of training examples

% You need to return the following variables correctly 
J = 0;   %这里只是保证程序可以正确运行，当定义代价函数后被覆盖。

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
predictions = X*theta; %predictions of training examples on all m examples
sqrErrors = (predictions - y).^2; %sqrErrors every Errors
J = 1/(2*m)*sum(sqrErrors);




% =========================================================================

end
