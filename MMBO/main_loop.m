% MULTISCALE MBO CODE

% This algorithm divides a data set into a number of classes (specified as n_classes)

% INPUT: data.txt, data_ground_truth.txt + fidelity.txt + parameters (see below)
% data.txt is a file that should be inputed as an N by n matrix where N is the number of elements and n is the number of attributes describing each element. Below is an example data set of three moons.
% data_ground_truth.txt is a file with correct class values in the range [1,n_classes]
% fidelity.txt is a file with 0 for non-labeled points and the class number for labeled points

%Example files for data.txt, data_ground_truth.txt and fidelity.txt is provided for the user.

% OUTPUT: the final classes of all elements, the accuracy 


% PARAMETERS
n_neighbors=6;     % number of nearest neighbors in the kNN computation
num_kernels=1;        % number of multiscale terms
iterNum = 400;  % number of iterations in energy minimization
dt= 0.003;          % parameter in the MBO method
power=[2];          % parameters in front of the multi-scale Laplacian terms (separated by semi-columns); there should be num_kernels parameters specified
linear_com= [1];     % parameters in front of the multi-scale Laplacian terms  (separated by semi-columns); there should be num_kernels parameters specified
C= 250;         % parameter in front of fidelity term
n_eigs= 200;       % number of eigenvectors to be computed
sigma= [1.25];     % parameters in weight function computation (separated by semi-columns); there should be num_kernels parameters specified



% LOADING OF DATA, LABELED VALUES AND GROUND TRUTH
data = textread('data.txt');
fidelity= textread(sprintf('fidelity.txt',i));
ground_truth = textread('data_ground_truth.txt'); % ground truth, must take integer values from 1 to n_classes


% CONSTRUCTION OF GRAPH AND COMPUTATION OF EIGENVECTORS
ns = createns(data,'nsmethod','kdtree');     
[idx, dist] = knnsearch(ns,data,'k',n_neighbors,'distance','euclidean');  % construction of the k-nearest neighbor graph
N= size(data,1); % number of data elements
n= size(data,2);  % number of attributes
n_classes=size(unique(ground_truth),1);  % number of classes
counter=size(find(fidelity==0),1); % counting the number of testing (non-labeled) elements
[Phi, Lambda] = Laplacian_eig_calculator(idx,dist,n_neighbors,N,sigma,power,linear_com,num_kernels,n_eigs); % computation of eigenvectors


% FIDELITY TERM 
% lam=1 for labeled points and lam=0 otherwise
lam = zeros(N,1);
for i=1:N
   if (fidelity(i,1)>0)
       lam(i)=1;
   end
end


% INITIALIZATION (RANDOM)
u0= rand(N,n_classes);
u0= projection_to_simplex(u0);
for i=1:N
    if (lam(i,1)==1)
        u0(i,:)= zeros(1,n_classes);
        u0(i,ground_truth(i,1))=1;
    end
end
        

% MAIN ALGORITHM
u = MBO(u0, Lambda, Phi, lam, C, dt, iterNum);


% EVALUATION OF THE METHOD
accuracy=0;
final_class=zeros(N,1);
for i=1:N
    if (lam(i,1)==0)
      y= u(i,:);
      [~,d]= max(y);
       final_class(i,1)=d;
       if ((ground_truth(i,1)==final_class(i,1)))
           accuracy= accuracy+1;
       end
    end
     if (lam(i,1)==1)
       final_class(i,1)=ground_truth(i,1);
     end
 end
 accuracy=accuracy/counter*100;
 
 
 % PRINTING THE ACCURACY
 fprintf('For this labeled set, the accuracy is %f percent. \n', accuracy);
