function [OUT] = testpcn(X,W)
% =====================================================
% testpcn - obt�m a sa�da do perceptron treinado
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
% =====================================================
% par�metros de sa�da
% =====================================================
%(1) OUT : matriz N x n com as sa�das fornecidas pelo perceptron
% =====================================================

% add extra component per input vector to account for the bias
X = cat(2,-ones(size(X,1),1),X);

OUT = X*W;    
OUT(logical(OUT>0))=1;
OUT(logical(OUT<0))=0;

end

