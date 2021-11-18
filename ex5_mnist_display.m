% =====================================================
% Prof. Saraiva, 2020/1
% =====================================================
% Ilustra o reconhecimento de dígitos usando o perceptron 
% treinado para identificar sequências de 11 dígitos. 
% OBS: o CPF tem 11 dígitos.
% =====================================================
clear; clc; close all; 

% carrega matriz de pesos sinápticos do perceptron treinado
% load W_717_10; 
load W_784_10; 

% número de neurônios
n = size(W,2);

% carrega conjunto de teste
load 'MNIST_testSet.mat'; 

% número de componentes por padrão de entrada
m = size(TEST,1)*size(TEST,2); 

% número de padrões de entrada no conjunto de teste
M = size(TEST,3); 
 
% monta a matriz de teste
testSet = zeros(m,M);
for i=1:M
    testSet(:,i) = reshape(TEST(:,:,i)',m,1);
end;

% alvos (rótulos) do conjunto de teste
testLabels = zeros(n,M);
for j=1:M    
    testLabels(TESTLABELS(j)+1,j)=1;
end;

% aplica seleção de atributos (se for o caso)
if(~isempty(IDX)>0)
    testSet=testSet(IDX,:);
end;    

% normaliza os valores de luminância - originalmente valores inteiros
% no intervalo [0,255] - para valores reais no intervalo [0,1]
MAX = max(max(testSet));
MIN = min(min(testSet));
testSet = (MAX-testSet)./(MAX-MIN);

% testa o perceptron treinado
OUT = testpcn(testSet',W)';

%inicializa índice dos subplots
nplot = 1;

% numero de sequencias de 11 dígitos
nseq = 4;

% inicializa contador do número de erros
nerror = 0;

for k=1:nseq
    
    % seleciona 11 dígitos aleatoriamente no conjunto de teste
    IDX = randperm(M);
    IDX = IDX(1:11);
    
    for i=1:11
    
        subplot(nseq,11,nplot);
        
        % exibe o dígito manuscrito
        imshow(imcomplement(TEST(:,:,IDX(i))'));
        
        % conversao de one-hot encoding para inteiro
        POS = find(OUT(:,IDX(i)));
        
        if(length(POS)>1) % se mais de um neurônio disparou    
            POS=min(POS);
        elseif(isempty(POS)) % se nenhum neurônio disparou
            POS=1;
        end;
        
        if((POS-1)==TESTLABELS(IDX(i)))
            COR='black';
        else
            COR='red';
            nerror = nerror+1;
        end;
        
        set(gca,'FontSize',35);                
        % POS-1, pois a posição "1" corresponde ao "0"
        title(num2str(POS-1),'Color',COR);    
        
        % atualiza índice do subplot
        nplot = nplot+1; 
        
    end;
end;

% acurácia do classificador (considerando a nseq sequências de
% 11 dígitos)
disp(100*(11*nseq-nerror)/(11*nseq));

