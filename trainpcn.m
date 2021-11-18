function [W,H,E] = trainpcn(W,X,T,lr,ne,tipo,options)
% =====================================================
% trainpcn - treinamento supervisionado do perceptron
% Prof. Saraiva, 2019/1
% =====================================================
% W : matriz inicial de pesos da rede (matriz m x n)
% W = [w_ij], i = 1,..,m; j=1,..,n
% m = n�mero de n�s de entrada
% n = n�mero de neur�nios
% w_ij � o peso entre o i-�simo n� de entrada e o j-�simo neur�nio
% -----------------------------------------------------
% X : matriz de treinamento (matriz N x m)
% X = [x_ij], i=1,..,N, j=1,..,m
% N = n�mero de vetores de entrada
% -----------------------------------------------------
% T : matriz alvo (matriz N x n)
% T = [t_ij], i=1,..,N, j=1,..,m
% -----------------------------------------------------
% lr: taxa ou passo de aprendizado (valor entre 0 e 1)
% -----------------------------------------------------
% ne: n�mero de �pocas no treinamento
% -----------------------------------------------------
% tipo: [0] batch [1] sequencial
% -----------------------------------------------------
% options: [0] n�o [1] sim
% options(1) - exibe epoca
% options(2) - shuffle input patterns at the beginning of each epoch 
% =====================================================
% par�metros de sa�da
% =====================================================
%(1) W : matriz final de pesos da rede 
%(2) H : cell array com o hist�rico dos pesos da rede ao longo das �pocas
% =====================================================

% add extra component per input vector to account for the bias
X = cat(2,-ones(size(X,1),1),X);

N = size(X,1); % n�mero de padr�es de treinamento
n = size(T,2); % n�mero de neur�nios

% classifica��es erradas por �poca    
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
        % calcula sa�da do perceptron
        output = X*W;         
        output(logical(output>0))=1;
        output(logical(output<0))=0;
        % regra de aprendizado do perceptron 
        % atualiza matriz de pesos
        W = W + lr*X'*(T-output);            
    elseif tipo==1 % treinamento sequencial                   
        for i = 1:N             
            % calcula sa�da do perceptron
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

