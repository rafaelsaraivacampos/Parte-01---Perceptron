% =====================================================
% logicGates - Prof. Saraiva, 2019/1
% =====================================================
% - Utiliza perceptron com um �nico neur�nio para simular
% as portas l�gicas  AND, OR, NAND, NOR, XOR
% - Este � um exemplo meramente did�tico: obviamente
% ningu�m vai efetivamente utilizar um perceptron para essa
% finalidade (simular portas l�gicas); o objetivo aqui � 
% ilustrar que o perceptron pode resolver apenas problemas 
% lineares (o que fica evidenciado diante da impossibilidade 
% de construir uma porta XOR)
% =====================================================
clearvars; clc; close all;

% padr�es de treinamento
X = [0 0;0 1;1 0;1 1];

% alvos 
T{1} = [0;0;0;1]; % AND
T{2} = [1;1;1;0]; % NAND
T{3} = [0;1;1;1]; % OR
T{4} = [1;0;0;0]; % NOR
T{5} = [0;1;1;0]; % XOR

% t�tulos dos sub-gr�ficos
TITLE = {'AND','NAND','OR','NOR','XOR'};

lr = .4; % taxa de aprendizado
ne = 15; % n�mero de �pocas

% n�mero de padr�es de entrada
N = size(X,1);

% n�mero de n�s de entrada (bias node inclu�do)
m = size(X,2)+1;

% n�mero de neur�nios
n = size(T{1},2);

for p=1:5

    subplot(2,3,p);
    
    % inicializa matriz de pesos com valores aleat�rios e pequenos
    W = normrnd(0,0.01,[m n]);

    % treina o perceptron
    W = trainpcn(W,X,T{p},lr,ne,1,[0 0]);

    % plota padr�es de entrada
    for i=1:N
        if(T{p}(i)==0)
            cor = 'r';
        else
            cor='b';
        end;
        scatter(X(i,1),X(i,2),cor,'+');
        hold on; grid on;
    end;
    
    xlabel('x_1'); ylabel('x_2'); 
    title(TITLE{p});

    % plota limiar de decis�o
    x=-.5:1.5;
    plot(x,-(W(2)*x-W(1))/W(3),'k');
    ylim([-.5 1.5]); xlim([-.5 1.5]);
    pbaspect([1 1 1]);

    % testa o perceptron treinado
    disp(TITLE{p});
    disp(testpcn(X,W)');
    
end;


