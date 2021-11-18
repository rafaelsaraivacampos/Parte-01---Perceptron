% =====================================================
% Prof. Saraiva, 2020/1
% =====================================================
% mnist - utilizando o perceptron para reconhecimento
% de padr�es - identifica��o de d�gitos (0 a 9) manuscritos
% Prof. Saraiva, 2019/1
% =====================================================
clear; clc; close all; 

% define semente do gerador de n�meros aleat�rios
rng(999);

% ATEN��O: executar primeiro os scripts para conversao
% da base MNIST para formato tabular (readsMNISTtestSet e
% readsMNISTtrainSet)
load 'MNIST_trainSet.mat';
load 'MNIST_testSet.mat';

FEATURE_SELECTION = 1; % [0] todos os atributos; [1] com  sele��o de atributos; 

TIPO = 1; % tipo de treinamento ([0] lote [1] sequencial)
lr = 0.25; % taxa de aprendizado
epc = 19; % n�mero de �pocas
n = 10; % n�mero de neur�nios (1 por d�gito:0,1,3,...,9)

% cada padr�o de treinamento � um vetor com 784 componentes
% cada componente corresponde � lumin�ncia de um pixel de uma
% imagem digitalizada de um d�gito manuscrito, com resolu��o 28 x 28 pixels

% n�mero de padr�es de treinamento
N = size(TRAIN,3); 

% n�mero de componentes por padr�o de entrada
m = size(TRAIN,1)*size(TRAIN,2); 

% monta a matriz de treinamento que ser� fornecida � fun��o trainpcn
trainSet = zeros(m,N);
for i=1:N
    trainSet(:,i) = reshape(TRAIN(:,:,i)',m,1);
end;

% n�mero de padr�es de entrada no conjunto de teste
M = size(TEST,3); 
 
% monta a matriz de teste
testSet = zeros(m,M);
for i=1:M
    testSet(:,i) = reshape(TEST(:,:,i)',m,1);
end;

% alvos (r�tulos) do conjunto de treinamento
trainLabels = zeros(n,N);
for j=1:N    
    trainLabels(TRAINLABELS(j)+1,j)=1;
end;

% alvos (r�tulos) do conjunto de teste
testLabels = zeros(n,M);
for j=1:M    
    testLabels(TESTLABELS(j)+1,j)=1;
end;

% ===================================================
% PRESERVA APENAS AS VARI�VEIS COM VARI�NCIA N�O-NULA
% ===================================================    
if(FEATURE_SELECTION)
    
    VAR = std(trainSet');
    IDX = find(VAR>0);
    
    subplot(1,2,1);
    imagesc(reshape(VAR,size(TRAIN,1),size(TRAIN,2)));
    pbaspect([1 1 1]); colormap('hot'); title('(a)'); colorbar;
    set(gca,'FontSize',18);
    
    subplot(1,2,2);
    AUX = ones(length(VAR),1); AUX(IDX)=0;
    imagesc(imcomplement(reshape(AUX,size(TRAIN,1),size(TRAIN,2))));
    pbaspect([1 1 1]); title('(b)'); colorbar; grid on;
    set(gca,'FontSize',18);
    
    trainSet = trainSet(IDX,:);
    testSet = testSet(IDX,:);
    m = size(trainSet,1); % atualiza o n�m. de comp. por padr�o de entrada
    
else
    IDX=[];
end;

% normaliza os valores de lumin�ncia - originalmente valores inteiros
% no intervalo [0,255] - para valores reais no intervalo [0,1]
MAX = max(max([trainSet,testSet]));
MIN = min(min([trainSet,testSet]));
trainSet = (MAX-trainSet)./(MAX-MIN);
testSet = (MAX-testSet)./(MAX-MIN);

% inicializa matriz de pesos com valores aleat�rios e pequenos
W = normrnd(0,0.01,[m+1 n]);

% treinamento supervisionado do perceptron
[W,H,E]=trainpcn(W,trainSet',trainLabels',lr,epc,TIPO,[1 1]);

% plota erro ao longo das �pocas
% (n�mero de classifica��es erradas x �poca)
figure; plot(E,'LineWidth',1.5,'Color','b'); grid on;
title('Classifica��es Erradas por �poca');
set(gca,'FontSize',14);
xlim([1 epc]);

% testa o perceptron treinado
OUT=testpcn(testSet',W)';

% exibe a matriz de confus�o do classificador
figure; plotconfusion(testLabels,OUT);

% salva o estado final da matriz de pesos sin�pticos
save(strcat('W_',num2str(size(trainSet,1)),'_',num2str(n)),'W','IDX');



