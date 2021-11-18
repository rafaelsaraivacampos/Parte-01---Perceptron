function [W,H,E] = trainpcn(W,X,T,lr,ne,tipo,options)
% =====================================================
% trainpcn - treinamento supervisionado do perceptron
% Prof. Saraiva, 2019/1
% =====================================================
% W : matriz inicial de pesos da rede (matriz m x n)
% W = [w_ij], i = 1,..,m; j=1,..,n
% m = número de nós de entrada
% n = número de neurônios
% w_ij é o peso entre o i-ésimo nó de entrada e o j-ésimo neurônio
% -----------------------------------------------------
% X : matriz de treinamento (matriz N x m)
% X = [x_ij], i=1,..,N, j=1,..,m
% N = número de vetores de entrada
% -----------------------------------------------------
% T : matriz alvo (matriz N x n)
% T = [t_ij], i=1,..,N, j=1,..,m
% -----------------------------------------------------
% lr: taxa ou passo de aprendizado (valor entre 0 e 1)
% -----------------------------------------------------
% ne: número de épocas no treinamento
% -----------------------------------------------------
% tipo: [0] batch [1] sequencial
% -----------------------------------------------------
% options: [0] não [1] sim
% options(1) - exibe epoca
% options(2) - shuffle input patterns at the beginning of each epoch 
% =====================================================
% parâmetros de saída
% =====================================================
%(1) W : matriz final de pesos da rede 
%(2) H : cell array com o histórico dos pesos da rede ao longo das épocas
% =====================================================

% add extra component per input vector to account for the bias
X = cat(2,-ones(size(X,1),1),X);

N = size(X,1); % número de padrões de treinamento
n = size(T,2); % número de neurônios

% classificações erradas por época    
E = zeros(ne,1);

for epoca = 1:ne        

    if(options(1))
        disp(epoca);        
    end;

    if(options(2)==1)
        IDX = randperm(N);
        X = X(IDX,:);
        T = T(IDX,:);
    end;

    H{epoca} = W;        

    if tipo==0 % treinamento de lote ("batch")
        % calcula saída do perceptron
        output = X*W;         
        output(logical(output>0))=1;
        output(logical(output<0))=0;
        % regra de aprendizado do perceptron 
        % atualiza matriz de pesos
        W = W + lr*X'*(T-output);            
    elseif tipo==1 % treinamento sequencial                   
        for i = 1:N             
            % calcula saída do perceptron
            output = logical((X(i,:)*W)>0);            
            % regra de aprendizado do perceptron 
            % atualiza matriz de pesos            
            W = W + lr*X(i,:)'*(T(i,:)-output);                
        end
    end;

    OUT = X*W;
    OUT(logical(OUT>=0))=1;
    OUT(logical(OUT<0))=0;        
    E(epoca)=length(find(sum(abs(OUT-T),2)));
    
end;

