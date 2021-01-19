function [Ktilde,K_test_tilde]=Deform_hermite(r,K,A,options,K_test)
% Computes the semi-supervised Kernel
% [Ktilde,K_test_tilde]=Deform(r,K,M,K_test)
% Inputs:
% K: the gram matrix of a kernel over labeled+unlabeled data (nxn matrix)
% M: a graph regularizer (nxn  matrix)
% K_test: the gram matrix of a kernel between training and test points
% (optional) size m x n for m test points.
% r: deformation ratio (gamma_I/gamma_A)
% Outputs:
% Ktilde: the gram matrix of the semi-supervised deformed kernel over labeled+unlabeled
% data
% K_test_tilde: the gram matrix of the semi-supervised deformed kernel
% between training and test points

I=eye(size(K,1));

%% number of hermite graphs
n = size(A,3);
%% Constructing a graph
for i=1:n
	if i == 1
		M = A(:,:,1)*options(1).coeff;
	else
		M = M + A(:,:,i)*options(i).coeff;
	end
end
%M = M^options(1).LaplacianDegree;

Ktilde=(I+r*K*M)\K;
%Ktilde = K - K*(I+r*M1*K)\M1*K';
if exist('K_test','var')
    K_test_tilde=(K_test - r*K_test*M*Ktilde);
end

