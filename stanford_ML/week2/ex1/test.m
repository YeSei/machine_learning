data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
fprintf(mu);
fprintf(sigma);
mu = mean(X);     
sigma = std(X);
fprintf(mu);
fprintf(sigma);

[X mu sigma] = featureNormalize(X);
