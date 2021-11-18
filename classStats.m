function [S]=classStats(targets,outputs)
% =====================================================
% classStats - estat�sticas da performance do classificador
% Prof. Saraiva, 2019/1
% =====================================================
% targets : matriz de alvos (matriz Nc x N)
% N = n�mero de vetores de sa�da
% Nc = n�mero de classes
% targets = [t_ij], i = 1,..,Nc; j=1,..,N
% t_ij = 0, se a classe-alvo do k-�simo vetor � dif. de i
% t_ij = 1, se a classe-alvo do k-�simo vetor � igual a i
% -----------------------------------------------------
% outputs : matriz de sa�das da rede neural (matriz Nc x N)
% outputs = [o_ij], i = 1,..,Nc; j=1,..,N
% o_ij = 0, se a classe de sa�da do k-�simo vetor � dif. de i
% o_ij = 1, se a classe de sa�da do k-�simo vetor � igual a i
% =====================================================
% par�metros de sa�da
% =====================================================
% S: estrutura com os seguintes campos
% acur = acur�cia do classificador
% prec = precisao por classe
% espec = especificidade por classe
% recall = sensibilidade (recall) por classe
% =====================================================

[c,cm] = confusion(targets,outputs);

Nc = size(cm,1); % n�mero de classes
N = size(outputs,2); % n�mero de padr�es de sa�da

TP = zeros(Nc,1);
FP = zeros(Nc,1);
TN = zeros(Nc,1);
FN = zeros(Nc,1);

espec = zeros(Nc,1);
prec = zeros(Nc,1);
recall = zeros(Nc,1);

for i=1:Nc    
    
    TP(i)=cm(i,i);
    FN(i)=sum(cm(i,:))-TP(i);
    FP(i)=sum(cm(:,i))-TP(i);
    TN(i)=N-TP(i)-FP(i)-FN(i);

    % precis�o = (classifica��es positivas corretas)/(total classifica��es positivas)
    prec(i) = TP(i)/(TP(i)+FP(i));
        
    % sensibilidade = (classifica��es positivas corretas)/(total de inst�ncias positivas)
    recall(i) = TP(i)/(TP(i)+FN(i));
    
    % especificidade = (classifica��es negativas corretas)/(total de inst�ncias negativas)
    espec(i) = TN(i)/(TN(i)+FP(i));
    
    
end;    

S.acur = trace(cm)/sum(sum(cm));
S.prec = prec;
S.recall = recall;
S.espec = espec;






