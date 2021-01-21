
function u  = MBO(u0, Lambda, Phi, lam, C, dt, iterNum)

u = u0;
a = Phi'*u; 
b = zeros(size(Phi,2),size(u,2));
N_s=3;
Denom = 1 + (dt/N_s)*Lambda;



% MAIN ITERATION
for n = 1:iterNum
    

    % APPLYING THE MULTISCALE HEAT EQUATION N_s TIMES
    for i=1:N_s
        for j=1:size(u,2)
            a(:,j) = (a(:,j) - (dt/N_s)*b(:,j))./Denom;
        end
    u = Phi*a;     
    b = C*(Phi'*(lam.*(u-u0)));  
    end
    
    % PROJECTION ONTO SIMPLEX
    u= projection_to_simplex(u);

    % DISPLACEMENT
    u= real(projection_to_vertex(u));
    
        
end


