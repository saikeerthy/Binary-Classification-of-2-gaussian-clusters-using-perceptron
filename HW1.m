clear all
close all
% for repeatability
rng(5);
% Creating training data points
% gaussian cluster with center (0,0) and variance 1 k1 = randn(1000,2);
% gaussian cluster with center (2.5,0) and variance 1 k2 = [randn(1000,1)+2.5 randn(1000,1)];
% input data
X = [k1(:,1)' k2(:,1)';k1(:,2)' k2(:,2)'];
% target data
T = [ ones(1,1000) zeros(1,1000)];
plotpv(X,T);
% perceptron network
net = perceptron;
hold on
% drawing a line with weight IW and bias b
linehandle = plotpc(net.IW{1},net.b{1});
for a = 1:7
% neural network training and adjusting the decision boundary
[net,Y,E] = adapt(net,X,T);
linehandle = plotpc(net.IW{1},net.b{1},linehandle); drawnow;
end;
figure;
% plotting the confusion matrix of training data plotconfusion(T,Y,'confusion matrix of train data') view(net)
% for repeatability
rng(1);
% creating test data points
k11 = randn(1000,2);
k22 = [randn(1000,1)+2.5 randn(1000,1)];
x = [k11(:,1)' k22(:,1)';k11(:,2)' k22(:,2)'];
y = net(x);
% plotting the confusion matrix of test data figure;
plotconfusion(T,y, 'confusion matrix of test data')
