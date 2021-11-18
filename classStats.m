function [S]=classStats(targets,outputs)
% =====================================================
% classStats - estatísticas da performance do classificador
% Prof. Saraiva, 2019/1
% =====================================================
% targets : matriz de alvos (matriz Nc x N)
% N = número de vetores de saída
% Nc = número de classes
% targets = [t_ij], i = 1,..,Nc; j=1,..,N
% t_ij = 0, se a classe-alvo do k-ésimo vetor é dif. de i
% t_ij = 1, se a classe-alvo do k-ésimo vetor é igual a i
% -----------------------------------------------------
% outputs : matriz de saídas da rede neural (matriz Nc x N)
% outputs = [o_ij], i = 1,..,Nc; j=1,..,N
% o_ij = 0, se a classe de saída do k-ésimo vetor é dif. de i
% o_ij = 1, se a classe de saída do k-ésimo vetor é igual a i
% =====================================================
% parâmetros de saída
% =====================================================
% S: estrutura com os seguintes campos
% acur = acurácia do classificador
% prec = precisao por classe
% espec = especificidade por classe
% recall = sensibilidade (recall) por classe
% =====================================================

[c,cm] = confusion(targets,outputs);

Nc = size(cm,1); % número de classes
N = size(outputs,2); % número de padrões de saída

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

    % precisão = (classificações positivas corretas)/(total classificações positivas)
    prec(i) = TP(i)/(TP(i)+FP(i));
        
    % sensibilidade = (classificações positivas corretas)/(total de instâncias positivas)
    recall(i) = TP(i)/(TP(i)+FN(i));
    
    % especificidade = (classificações negativas corretas)/(total de instâncias negativas)
    espec(i) = TN(i)/(TN(i)+FP(i));
    
    
end;    

S.acur = trace(cm)/sum(sum(cm));
S.prec = prec;
S.recall = recall;
S.espec = espec;






