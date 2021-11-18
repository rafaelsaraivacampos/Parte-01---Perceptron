% =====================================================
% Prof. Saraiva, 2020/1
% =====================================================
% idem exemplo 3a (gaussianas), porém usando one-hot encoding
% =====================================================
clearvars; clc; close all;

% controla o gerador de números aleatórios
% (usado na inicialização das matrizes de pesos)
% para fins de reproducibilidade
seed=111; rng(seed);

ne = 100; % número de épocas
lr = 0.15; % taxa de aprendizado
TIPO = 0; % [0] lote(batch) [1] sequencial
P = .75; % percentagem para treinamento
HISTORICO = 0; % [0] não [1] sim

% -------------------------
% padroes de entrada
% -------------------------
N = 200; % padrões por classe
DP = [.2 .6 .4]; % desvio padrão de cada gaussiana
C = [-2 2;0 -2;2 2]; % posição central de cada distribuição
% exemplo onde one-hot encoding não funciona
%C = [-2 -2;0 0;2 2]; % posição central de cada distribuição

% gera os padrões de entrada (N por classe)
nc = length(DP); % número de classes
X=zeros(nc*N,2); 
for i=1:nc
    X((i-1)*N+1:N*i,:) = normrnd(0,DP(i),[N 2])+C(i,:);
end;

% -------------------------
% define targets
% -------------------------
T(1:N,:) = cat(2,ones(N,1),zeros(N,2));
T(N+1:2*N,:) = cat(2,zeros(N,1),ones(N,1),zeros(N,1));
T(2*N+1:3*N,:) = cat(2,zeros(N,2),ones(N,1));

% -------------------------
% initialize weigths with random (and small) values
% -------------------------
m = size(X,2)+1; % número de nós de entrada
n = size(T,2); % número de neurônios
W = normrnd(0,0.01,[m n]);

% -------------------------
% builds train and test sets
% -------------------------
IDX = randperm(nc*N);
IDX1 = IDX(1:P*(nc*N));
IDX2 = IDX(P*(nc*N)+1:length(IDX));
trainSet = X(IDX1,:);
testSet = X(IDX2,:);
testSetSize = length(IDX2);

% -------------------------
% plota padrões de entrada por classe 
% (CONJUNTO DE TREINAMENTO)
% -------------------------
COR={'b','r','k'};
subplot(2,2,1);
for i=1:nc
    I = find(T(IDX1,i)==1);
    scatter(trainSet(I,1),trainSet(I,2),COR{i},'+');
    hold on;
end;
grid on;
xlabel('x_1'); ylabel('x_2'); 
set(gca,'FontSize',14);
%xlim([-4 4]); 
ylim([-4 4]);
pbaspect([1 1 1]);
title('Conjunto de treinamento');
legend('C1','C2','C3','Location','se');

% -------------------------
% train the perceptron
% -------------------------
[W,H,E] = trainpcn(W,trainSet,T(IDX1,:),lr,ne,TIPO,[0 0]);

% -------------------------
% plota as fronteiras entre as classes
% w1*x1 + w2*x2 + w3 = 0 (w3 is the threshold, i.e., (-1)*bias)
% -------------------------
x=-4:4; COR={'g','y'};
for i=1:ne 
    for j=1:n
        if(i<ne)
            if(HISTORICO)
                plot(x,-(H{i}(2,j)*x-H{i}(1,j))/H{i}(3,j),...
                    'Color',COR{j},'LineStyle','--');                
                getframe;
            end;
        else
            plot(x,-(W(2,j)*x-W(1,j))/W(3,j),...
                'Color','k','LineStyle','-');
        end;
    end;
end;    

% -------------------------
% plota padrões de entrada por classe 
% (CONJUNTO DE TESTE)
% -------------------------
subplot(2,2,2);
COR={'b','r','k'};
for i=1:nc
    I = find(T(IDX2,i)==1);
    scatter(testSet(I,1),testSet(I,2),COR{i},'+');
    hold on;
end;
grid on;
xlabel('x_1'); ylabel('x_2'); 
set(gca,'FontSize',14);
%xlim([-4 4]); 
ylim([-4 4]);
pbaspect([1 1 1]);
title('Conjunto de teste');
legend('C1','C2','C3','Location','se');

% -------------------------
% plota as fronteiras entre as classes
% w1*x1 + w2*x2 + w3 = 0 (w3 is the threshold, i.e., (-1)*bias)
% -------------------------
x=-4:4; COR={'g','y'};
for i=1:ne 
    for j=1:n
      plot(x,-(W(2,j)*x-W(1,j))/W(3,j),'Color','k','LineStyle','-');
    end;
end;    

% -------------------------
% plota erro ao longo das épocas
% (número de classificações erradas x época)
% -------------------------
subplot(2,2,3:4); 
plot(E,'LineWidth',1.5,'Color','b'); 
grid on;
title('Classificações Erradas por Época');
xlim([1 ne]);
xlabel('Época'); ylabel('Erros'); 
set(gca,'FontSize',14);

% -------------------------
% matriz de confusão do classificador
% -------------------------
figure; plotconfusion(T(IDX2,:)',testpcn(testSet,W)');

