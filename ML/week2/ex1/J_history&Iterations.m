data = load('ex1data1.txt');
X = data(:, 1);
y = data(:, 2);
m = length(y); 
X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters
iterations = 1500;
alpha = 0.01;
[theta,J_history] = gradientDescent(X, y, theta, alpha, iterations);
%   测试函数返回两个参数，用列表接受。

plot(1:1500,J_history,'rx','MarkerSize',10);
%   将1-1500次迭代的代价函数标注出来；可以看到代价函数逐渐下降。

ylabel('cost function');
xlabel('iterations');