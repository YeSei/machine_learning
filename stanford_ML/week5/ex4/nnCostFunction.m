function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

%首先将输出单元y用矩阵表示出来：
I = eye(num_labels);   % 类别矩阵：用label x label矩阵表示，
Y = zeros(m, num_labels);  % 输出矩阵：表示成m x label 的矩阵形式。
for i=1:m
  Y(i, :)= I(y(i), :);
end
%然后我们要写出代价函数的代码。那么我们先利用正向传播把目标函数h写出来，再写正则化，最后拼在一起就好了。
%只有三层神经网络,先按照前向传播的步骤（笔记13有例子）写出各层激励：
a1 = [ones(m, 1) X];          %加上偏置单元
z2 = a1*Theta1';
a2 = [ones(size(z2, 1), 1) sigmoid(z2)];
z3 = a2*Theta2';
a3 = sigmoid(z3);
h = a3;

%正则化部分：
p = sum(sum(Theta1(:, 2:end).^2, 2)) + sum(sum(Theta2(:, 2:end).^2, 2));

%代价函数：
J = sum(sum((-Y) .* log(h) - (1-Y) .* log(1-h), 2))/m + lambda * p/(2 * m);


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

%接着我们要用后向传播去计算每一层激励的误差（笔记13）：
sigma3 = a3 - Y;
sigma2 = (sigma3 * Theta2) .* sigmoidGradient([ones(size(z2, 1), 1) z2]);
sigma2 = sigma2(:, 2:end);

%然后计算梯度：
delta_1 = (sigma2' * a1);
delta_2 = (sigma3' * a2); 
% 单个变量误差为一列，其梯度表示为sigma（L）.a(L-1)‘，多个变量时，记住行为输入变量个数m，列才是其误差。
% 梯度也表示为sigma（L）’*a（L-1），没行为其对应的单个样本的相应梯度。

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%最后把正则化加上：
p1 = (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
p2 = (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
Theta1_grad = delta_1./m + p1;
Theta2_grad = delta_2./m + p2;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
