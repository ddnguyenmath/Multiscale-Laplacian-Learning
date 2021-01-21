% this function computes the multiscale Laplacian and computes the
% extremal eigenvectors and eigenvalues.

function [V,D] = Laplacian_eig_calculator(idx,dist,n_neighbors,N,sigma,power,linear_com,num_kernels,n_eigs)

W= zeros(N,N,num_kernels);

for i=1:N
    for j=1:n_neighbors
           distance= dist(i,j);
        for k=1:num_kernels
            number= distance/sigma(k,1);
            number= number.^power(k,1);
            W(i,idx(i,j),k)= exp(-number);
        end
    end
end


for i=1:size(W,1)
    for k=1:num_kernels
        W(i,i,k)=0;
    end 
end


D= sum(W,2);
L= zeros(N,N);

% computing the multiscale Laplacian
for k=1:num_kernels
   L= L+ linear_com(k,1)*(diag(D(:,:,k))-W(:,:,k));
end


[V,D]= eigs(L,n_eigs,'sa');

V= real(V);
D= diag(D);

end

