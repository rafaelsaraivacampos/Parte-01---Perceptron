% =====================================================
% XOR - Prof. Saraiva, 2019/1
% =====================================================
% Ilustra alternativas para simular a porta XOR utilizando perceptrons:
% a) acrescenta mais uma componente no vetor de entrada:
% as classes de sa�da (0 ou 1) n�o s�o linearmente separ�veis
% em 2 dimens�es, mas o s�o em 3;
% b) acrescenta um neur�nio: com 2 neur�nios, o perceptron
% pode ter 4 classes de sa�da
% =====================================================
clearvars; clc; close all;

% alvos (porta XOR)
T{1} = [0;1;1;0];  % (a)
T{2} = [0 0;1 0;1 0;1 1]; % (b)

N = 4; % n�mero de padr�es de entrada
ne = 10; % n�mero de �pocas
lr = .4; % taxa de aprendizado

% ---------------------------------------------
% a) acrescenta uma componente no vetor de entrada:
% as classes de sa�da (0 ou 1) n�o s�o linearmente separ�veis
% em 2 dimens�es, mas o s�o em 3;
% ---------------------------------------------

% padr�es de treinamento
X = [0 0 1;0 1 0;1 0 0;1 1 1];

% n�mero de n�s de entrada (bias node inclu�do)
m = size(X,2)+1;

% n�mero de neur�nios
n = size(T{1},2);

% inicializa matriz de pesos com valores aleat�rios e pequenos
W = normrnd(0,0.01,[m n]);

% treina o perceptron
W = trainpcn(W,X,T{1},lr,ne,0,[0 0]);

% plota padr�es de entrada
subplot(1,2,1);
scatter3(X(1,1),X(1,2),X(1,3),'r','+');
hold on; grid on;
scatter3(X(2,1),X(2,2),X(2,3),'b','+');
scatter3(X(3,1),X(3,2),X(3,3),'b','+');
scatter3(X(4,1),X(4,2),X(4,3),'r','+');
xlabel('x_1'); ylabel('x_2');

   
% plota limiares de decis�o
[x,y]=meshgrid(-.5:.5:1.5);
mesh(x,y,(-W(2)*x-W(3)*y+W(1))/W(4),'EdgeColor','none',...
    'FaceColor','interp');
ylim([-.5 1.5]);
xlim([-.5 1.5]);
zlim([-.5 1.5]);
pbaspect([1 1 1]);
title('(a) acrescenta uma componente ao vetor de entrada');

% testa o perceptron treinado
disp(testpcn(X,W)');

% ---------------------------------------------
% b) acrescenta um neur�nio: com 2 neur�nios, o perceptron
% pode ter 4 classes de sa�da
% ---------------------------------------------

% padr�es de treinamento
X = [0 0;0 1;1 0;1 1];

% n�mero de n�s de entrada (bias node inclu�do)
m = size(X,2)+1;

% n�mero de neur�nios
n = size(T{2},2);

% inicializa matriz de pesos com valores aleat�rios e pequenos
W = normrnd(0,0.01,[m n]);

% treina o perceptron
W = trainpcn(W,X,T{2},lr,ne,1,[0 0]);

% plota padr�es de entrada
subplot(1,2,2);
scatter(X(1,1),X(1,2),'r','+');
hold on; grid on;
scatter(X(2,1),X(2,2),'b','+');
scatter(X(3,1),X(3,2),'b','+');
scatter(X(4,1),X(4,2),'r','+');
xlabel('x_1'); ylabel('x_2');
title('(b) acrescenta um neur�nio');

% plota limiares de decis�o
x=-.5:1.5;
for i=1:n
    plot(x,-(W(2,i)*x-W(1,i))/W(3,i),'k');
end;    
ylim([-.5 1.5]);
xlim([-.5 1.5]);

% testa o perceptron treinado
disp(testpcn(X,W)');

    