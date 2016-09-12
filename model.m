%% Machine Learning Assignment
%% ================ Part 1: initialization ================

% Clear and Close Figures
clear ; close all; %clc

fprintf('Loading data ...\n');

% Load Data
traindata = csvread('train_data.csv',1,1);
testdata = csvread('test_data.csv',1,1);

X = traindata(:,1:end-1); y = traindata(:,end);
m = length(y);
%row = size(X,1);
col = size(X,2);

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

X = [X X(:,1:col).^2];
testdata = [testdata, testdata(:,:).^2];

[X, mu, sigma] = featureNormalize(X);
[testdata] = featureNormalize(testdata);

% Add intercept term to X
X = [ones(m, 1) X];
testdata = [ones(length(testdata), 1) testdata];

%% ================ Part 2: Gradient Descent and Output ================
% Choose parameters
alpha = 0.01;
num_iters = 2000;
lambda = [-200 -175 -150 -125 -100 -90 -80 -70 -60 -10 1];

tr_rmse = zeros(1,length(lambda));
cv_rmse = zeros(1,length(lambda));

%random permutation
permu = randperm(m);
X = X(permu,:);
y = y(permu,:);

%divide into training cross validation and test set
training_size = floor(0.8 * m);
cv_size = floor(0.1 * (m-training_size));

X_train = X(1:training_size,:);
X_cv = X(training_size+1:training_size+cv_size,:);
X_test = X(training_size+1+cv_size:end,:);

y_train = y(1:training_size,:);
y_cv = y(training_size+1:training_size+cv_size,:);
y_test = y(training_size+1+cv_size:end,:);

theta = zeros(size(X_train,2), length(lambda));
%% ================ Part 2: Gradient Descent ================

fprintf('Running gradient descent for different values of lambda ...\n');
for i= 1:length(lambda)
% Run Gradient Descent 
theta(:,i) = gradientDescentMulti(...
    X_train, y_train, theta(:,i), alpha, num_iters, lambda(i));

tr_rmse(i) = sqrt((1 / m) * ...
    (X_train * theta(:,i) - y_train)' * (X_train * theta(:,i) - y_train));
cv_rmse(i) = sqrt((1 / m) * ...
    (X_cv * theta(:,i) - y_cv)' * (X_cv * theta(:,i) - y_cv));
end
fprintf('train rmse\n');
fprintf('%f\t',tr_rmse);
fprintf('\ncv rmse\n');
fprintf('%f\t',cv_rmse);
[value, index] = min(cv_rmse);
fprintf('\nmin cv error is obtained for lambda: %f\n', lambda(index));
fprintf('test error is: %f\n', sqrt((1 / m) * ...
    (X_test * theta(:,index) - y_test)' * (X_test * theta(:,index) - y_test)));

theta = zeros(size(X,2), 1);
[theta, J_history] = gradientDescentMulti(...
    X, y, theta, alpha, num_iters, lambda(index));
output = testdata * theta;
testlen = length(testdata);
a = 0:testlen-1;
header = 'id,shares'; 
fid = fopen('output.csv','w');
fprintf(fid,'%s\n',header);
output = [a', output];
for k = 1:size(output,1)
    fprintf(fid,'%d,%.1f\n',output(k,:));
end
fclose(fid);
fprintf('output.csv file generated\n');