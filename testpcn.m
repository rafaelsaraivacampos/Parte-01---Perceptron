function [OUT] = testpcn(X,W)
% =====================================================
% testpcn - obtém a saída do perceptron treinado
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
% =====================================================
% parâmetros de saída
% =====================================================
%(1) OUT : matriz N x n com as saídas fornecidas pelo perceptron
% =====================================================

% add extra component per input vector to account for the bias
X = cat(2,-ones(size(X,1),1),X);

OUT = X*W;    
OUT(logical(OUT>0))=1;
OUT(logical(OUT<0))=0;

end

