% -----------------------
% CEFET-RJ Campus Petropolis
% Faculdade de Engenharia de Computação
% GCOM8056 - Sistemas Inteligentes - 2020/1
% Prof. Rafael Saraiva Campos
% -----------------------
clearvars; clc; close all;

% -------------------------
% controls random number generator
% (used in weigth matrix initialization)
% -------------------------
rng(2);

ne = 40; % number of epochs
lr = 0.25; % learning rate
TIPO = 1; % [0] batch [1] sequential
P = .85; % percentage of data used in training set
HISTORICO = 1; % [0] no [1] yes

% -------------------------
% Attribute Information:
% -------------------------
% 1. variance of Wavelet Transformed image (continuous)
% 2. skewness of Wavelet Transformed image (continuous)
% 3. curtosis of Wavelet Transformed image (continuous)
% 4. entropy of image (continuous)
% 5. class (integer)
% -------------------------
% uses UCI banknote authentication Data Set
% archive.ics.uci.edu/ml/datasets/banknote+authentication
% -------------------------
DATA = dlmread('data_banknote_authentication.txt');
X = DATA(:,1:4); % input data
N = size(DATA,1); % number of input vectors

% plot input variables
for i=1:4
    subplot(2,2,i);
    plot(DATA(:,i)); grid on; xlim([1 N]);
end;

% normalizes input data to 0 to +1 range
MIN = min(X);
MAX = max(X);
X = (X-MIN)./(MAX-MIN);
 
% -------------------------
% Initialize weigths with random (and small) values
% -------------------------
m = size(X,2)+1; % input nodes (bias node included)
n = 1; % number of neurons in the perceptron
W = normrnd(0,0.01,[m n]);

% -------------------------
% Builds train and test sets
% -------------------------
IDX = randperm(N);
IDX1 = IDX(1:floor(P*N));
IDX2 = IDX(floor(P*N)+1:N);
trainSet = X(IDX1,:);
testSet = X(IDX2,:);

% -------------------------
% Train the perceptron
% -------------------------
[W,H,E] = trainpcn(W,trainSet,DATA(IDX1,5),lr,ne,TIPO,[0 1]);

% -------------------------
% Plots classifier confusion matrix
% -------------------------
 plotconfusion(DATA(IDX2,5)',testpcn(testSet,W)');

% -------------------------
% Plots perceptron weigths per epoch
% -------------------------
figure;
for i=1:ne
    for j=1:5
       w(i,j)=H{i}(j);
    end;
end;

plot(w);
grid on;
legend('1st node','2nd node','3rd node','4th node','bias node');

% plots variables per class
figure; 

False = find(DATA(:,5)==1); 
True = find(DATA(:,5)==0);

TITLE = {'Variance','Skewness','Kurtosis','Entropy'};
for i=1:4
    subplot(2,2,i);
    plot(DATA(False,i),'r'); hold on; plot(DATA(True,i),'b');
    grid on;
    title(TITLE{i});
    legend('False','True');
end;    

% plota erro ao longo das épocas
% (número de classificações erradas x época)
figure;
plot(E,'LineWidth',1.5,'Color','b'); grid on;
title('Classificações Erradas por Época');
set(gca,'FontSize',14);
xlim([1 ne]);