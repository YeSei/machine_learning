function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
h=sigmoid(X*theta);
J = ((-y)'*log(h)-(1-y)'*log(1-h))/m;

grad = (X'*(h-y))/m;  

% 用X转置的原因理解，正常理解是，这里对0求偏导，每个0应该是所有x相应属性的偏导。
% 可以从维度来理解，grad是theta的每个0，这里X转置与theta维度相同



% =============================================================

end
