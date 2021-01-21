% this function projects a value on the Gibbs simplex to the nearest vertex
% of the simplex

function Y_projected = projection_to_vertex(Y)

    Y_projected= Y;
    N= size(Y,1);
    
for k=1:N
     y= Y(k,:); 
     n=size(y,2);
     [~,d]= max(y);
     Y_projected(k,:)= zeros(1,n);
     Y_projected(k,d)=1;
end
