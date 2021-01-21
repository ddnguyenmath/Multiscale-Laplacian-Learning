% this function projects a vector onto the Gibbs simplex.|
% The algorithm is taken from "Projection onto a simplex" by Y. Chen and X. Ye. 

function Y_projected = projection_to_simplex(Y)
   
    Y_projected= Y;
    N= size(Y,1);
    
    
for k=1:N
        
    y= Y(k,:); 
    n= size(y,2); 
    u= sort(y);
    i= n-1;
    
      
    while (i>0)        
        sum=0;
        for j=i+1:n
            sum= sum+ u(1,j);
        end
         t_i= (sum-1)/(n-i);  
         
         if (t_i> u(1,i))
            t_hat=t_i;
            i= -1;
         end
         
         i=i-1;     
    end
    
    
    if (i==0)
        sum=0;
        for j=1:n
            sum= sum+ u(1,j);
        end
        t_hat= (sum-1)/n;
    end
   
    
      u_projected= max(0, y-t_hat);
      Y_projected(k,:)= u_projected;
    
end

