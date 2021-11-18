% =====================================================
% Prof. Saraiva, 2020/1
% =====================================================
% um perceptron com um "n" neur�nios � treinado 
% supervisionadamente para classificar vetores de entrada 
% entre "nc" classes distintas
% =====================================================
clearvars; clc; close all;

% -------------------------
% controla o gerador de n�meros aleat�rios
% (usado na inicializa��o das matrizes de pesos)
% para fins de reproducibilidade
% -------------------------
seed=211; rng(seed);

lr = 0.15; % taxa de aprendizado
ne = 100; % n�mero de �pocas
P = .75; % percentagem para treinamento
TIPO = 0; % [0] lote(batch) [1] sequencial

% -------------------------
% op��o para exibir hist�rico de treinamento, ilustrando
% as fronteiras de decis�o ao longo do treinamento, al�m
% de armazenar os estados da matriz de pesos ao longo das �pocas
% de treinamento
% -------------------------
HISTORICO = 0; % [0] n�o [1] sim

% -------------------------
% gera os padr�es de entrada (N por classe)
% -------------------------
N = 200; % padr�es por classe
DP = [1 .6 .4]; % desvio padr�o de cada gaussiana
C = [-2 2;0 -2;2 2]; % po1si��o central de cada distribui��o

nc = length(DP); % n�mero de classes
X=zeros(nc*N,2); 
for i=1:nc
    X((i-1)*N+1:N*i,:) = normrnd(0,DP(i),[N 2])+C(i,:);
end;

% -------------------------
% define os alvos
% -------------------------
A = [0 0;0 1; 1 1]; % codigos binarios id. de cada classe
T(1:N,:) = repmat(A(1,:),N,1);
T(N+1:2*N,:) = repmat(A(2,:),N,1);
T(2*N+1:3*N,:) = repmat(A(3,:),N,1);

% -------------------------
% inicializa os pesos sin�pticos
% -------------------------
m = size(X,2)+1; % n�mero de n�s de entrada
n = size(T,2); % n�mero de neur�nios
W = normrnd(0,0.01,[m n]);

% -------------------------
% separa os conjuntos de treinamento e teste
% -------------------------
IDX = randperm(nc*N); % emebaralha os �ndices das linhas da matriz X
IDX1 = IDX(1:P*(nc*N)); % �ndices para o conj. de treinamento
IDX2 = IDX(P*(nc*N)+1:length(IDX)); % �ndices para o conj. de teste
trainSet = X(IDX1,:);
testSet = X(IDX2,:);
testSetSize = size(testSet,1);

% -------------------------
% plota o conjunto de treinamento
% -------------------------
subplot(2,3,1);
COR = {'b','r','k'};
for i=1:nc
    I = find((T(IDX1,1)==A(i,1))&(T(IDX1,2)==A(i,2)));
    scatter(trainSet(I,1),trainSet(I,2),COR{i},'+');
    hold on;
end;
grid on;
xlabel('x_1'); ylabel('x_2'); 
set(gca,'FontSize',14);
%xlim([-2 4]); 
ylim([-4 4]);
pbaspect([1 1 1]);
title('Conjunto de treinamento');
legend('C1','C2','C3','Location','se');

% -------------------------
% treinamento do perceptron
% -------------------------
[W,H,E] = trainpcn(W,trainSet,T(IDX1,:),lr,ne,TIPO,[0 0]);

% -------------------------
% plota as fronteiras de decis�o sobre o conj. de treinamento
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
% plota erro ao longo das �pocas
% (n�mero de classifica��es erradas x �poca)
% -------------------------
subplot(2,3,3); 
plot(E,'LineWidth',1.5,'Color','b'); 
grid on;
title('Classifica��es Erradas por �poca');
xlim([1 ne]);
xlabel('�poca'); ylabel('Erros'); 
set(gca,'FontSize',14);

% -------------------------
% plota o conjunto de teste
% -------------------------
subplot(2,3,2);
COR = {'b','r','k'};
for i=1:nc
    I = find((T(IDX2,1)==A(i,1))&(T(IDX2,2)==A(i,2)));
    scatter(testSet(I,1),testSet(I,2),COR{i},'+');
    hold on;
end;
grid on;
xlabel('x_1'); ylabel('x_2'); 
set(gca,'FontSize',14);
% xlim([-2 4]); 
ylim([-4 4]);
pbaspect([1 1 1]);
title('Conjunto de teste');
legend('C1','C2','C3','Location','se');

% -------------------------
% plota as fronteiras de decis�o sobre o conj. de teste
% w1*x1 + w2*x2 + w3 = 0 (w3 is the threshold, i.e., (-1)*bias)
% -------------------------
x=-4:4; COR={'g','y'};
for i=1:ne 
    for j=1:n
        plot(x,-(W(2,j)*x-W(1,j))/W(3,j),'Color','k','LineStyle','-');
    end;
end;    
 
% -------------------------------------
% Plota hist�rico dos pesos sin�pticos
% ------------------------------------- 
% if each cell contains the same type of data, 
% you can create a single variable by applying the array 
% concatenation operator, [], to the comma-separated list
V = [H{1:ne}];
for j=1:n % j-�simo neur�nio
    subplot(2,3,3+j);
    for i=1:m % i-�simo input node
        plot(V(i,j:2:n*ne));
        hold on; 
    end;
    grid on; xlim([1 ne]);
    xlabel('�poca'); ylabel('w'); 
    set(gca,'FontSize',14);
    pbaspect([1 1 1]);
    title(strcat('Neur�nio :',num2str(j)));
    legend(strcat('W0',num2str(j)),...
           strcat('W1',num2str(j)),...
           strcat('W2',num2str(j)),'Location','ne');
end;    

% -------------------------------------
% Sa�da do Perceptron 
% -------------------------------------
OUT = testpcn(testSet,W);

AUX=abs(OUT-T(IDX2,:));
disp(strcat('Acur�cia=',num2str(100*(1-length(find(AUX(:,1)+AUX(:,2)))/testSetSize))));



