function drawMLP(neuronsPerLayer)
% ================================
% drawMLP - plota topologia de um perceptron de múltiplas camadas
% Prof. Saraiva - 2019/1
% ================================
K=1;

% number of layers (including input layer)
L=length(neuronsPerLayer);

% include bias node (except in output layer)
neuronsPerLayer=neuronsPerLayer+ones(1,L);
neuronsPerLayer(L)=neuronsPerLayer(L)-1;

% max number of neurons in any given layer
N=max(neuronsPerLayer); 

% horizontal spacing between layers
r=1; dH = (N*2*r)/(L-1);

% vertical spacing between neurons on the same layer
dV = (N*2*r)./(neuronsPerLayer-1);

% if there is only one neuron on the output layer
FLAG=0;
if(sum(logical(isinf(dV)))>0)
    FLAG=1;
    dV(logical(isinf(dV)))=N*r; 
end;    
        
%% builts "center" struct array
for layer=1:L
    for neuron=1:neuronsPerLayer(layer)
        
        center(layer,neuron).x=dH*(layer-1);
        
        if(layer==FLAG*L) % otuput layer with just one neuron
            center(layer,neuron).y=dV(layer);            
        else % all other cases, for all layers
            center(layer,neuron).y=dV(layer)*(neuron-1);
        end;

    end;
end;

%% draws synapses
for layer=1:L-1
    for neuron=1:neuronsPerLayer(layer)
        
        % first neuron of all layers (except the output layer)
        % is a bias node (different color)
        if((layer<L)&&(neuron==1))
            LINECOLOR='r';
        % all other cases
        else
            LINECOLOR='k';
        end;   
        
        if(layer<L-1)
            % the nodes in layer l do not connect to the bias node in layer
            % l+1, so the mapping (see loop for below) starts in the second
            % node
            I=2;
        else
            I=1;
        end;
        
        % connects the current neuron (j) in the current layer(layer) 
        % to all nodes(except bias node) in the following layer(layer+1)
        for k=I:neuronsPerLayer(layer+1)
            line([center(layer,neuron).x center(layer+1,k).x],...
                [center(layer,neuron).y center(layer+1,k).y],'Color',LINECOLOR);        
            hold on;
        end;
        
    end;
end;    

%% draws input nodes, bias nodes and neurons
for layer=1:L
    for neuron=1:neuronsPerLayer(layer)
        
        % first neuron of all layers (except the output layer)
        % is a bias node (different color)
        if((layer<L)&&(neuron==1))
            COLOR='r';
        else
            % all other cases        
            COLOR='k';
        end;
        
        x0 = center(layer,neuron).x;  
        y0 = center(layer,neuron).y;        
        
        rectangle('Position',[x0-r,y0-r,2*r,2*r],...
            'Curvature',[1 1],'Facecolor',[1 1 1],'EdgeColor',COLOR);
        
        hold on;

    end;
end;

%% finish formatting the plot

pbaspect([1 1 1]);

xlim([-2*r 2*r*(N+1)]);
ylim([-2*r 2*r*(N+1)]);

title('MLP Topology');

set(gca,'YTickLabel','');
set(gca,'YTick',[]);

for i=2:L-1
    HLABEL{i}=i-1;
end;

HLABEL{1}='Input Layer';
HLABEL{L}='Output Layer';

set(gca,'XTick',[center(:,1).x]);
set(gca,'XTickLabel',HLABEL);
