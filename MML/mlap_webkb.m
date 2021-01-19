function results=mlap_webkb(dataset,options,joint,M) 
% [err_svm,err_rlsc]=ssl_hermite_kfold(dataset)
% dataset.mat should have matrices 
% X (n x d training data matrix, n examples, d features)
% Xt (test data matrix) (if available)     
% training labels Y      
% test labels Yt (if available)
%
% NN: number of nearest neighbors for graph laplacian computation
% SIGMA: width of RBF kernel
% gamma_A,gamma_I: the extrinsic and intrinsic regularization parameters
% DEGREE: power of the graph Laplacian to use as the graph regularizer
% M: Precomputed grah regularizer if available (n x n matrix).
% hermite_order: The highest order of hermite polynomial
%
% This code generates the mean results training and testing on data
% splits.
% 
% Data splits are specified as follows -- 
% folds: a n x R matrix with 0 or 1 values; each column indicates which
% examples are to be considered labeled (1) or unlabeled (0). 
% Alternatively, one can specify by a collection of indices in R x l matrix 
% idxLabs where each row gives the l indices for labeled examples. 
%
% For experiments in the paper "Beyond the Point Cloud", run e.g.,
% 
% Experiments('USPST'); % USPST.mat contains all the options
%
%  Vikas Sindhwani (vikass@cs.uchicago.edu)

setpaths


if ~exist('M','var') 
hermite_order = size(options,2)
for i=1:hermite_order
    if joint == 1
        load LINK_test.mat;
        M1 = full(laplacian_hermite(options(i),X));
        load PAGE_test.mat;
        M2 = full(laplacian_hermite(options(i),X));
        load PAGELINK_test.mat;
        M3 = full(laplacian_hermite(options(i),X));
        M  = (M1+M2+M3)/3;
        clear M1 M2 M3;
        M=M^options(i).LaplacianDegree;
    else
    load(dataset)
    M = full(laplacian_hermite(options(i),X));
    M=M^options(i).LaplacianDegree;
  end
  if i == 1
    A = M;
  else
    A = cat(3,A,M);
  end
end

tic;
% construct and deform the kernel
% K contains the gram matrix of the warped data-dependent semi-supervised kernel
G=calckernel(options(1),X); % calckernel needs kernel_type and kernel_param which defined by options[1]
r=options(1).gamma_I/options(1).gamma_A;
% the deformed kernel matrix evaluated over test data
fprintf(1,'Deforming Kernel\n');
if exist('Xt','var')
    Gt=calckernel(options,X,Xt);
    [K, Kt]=Deform(r,G,M,Gt);
else
    K=Deform_hermite(r,G,A,options);
end
% run over the random splits
for R=1:size(idxLabs,1)

    L=idxLabs(R,:); U=1:size(K,1); U(L)=[];
    data.K=K(L,L); data.X=X(L,:); data.Y=Y(L); % labeled data

    classifier_svmp=svmp(options(1),data);
    classifier_rlsc=rlsc(options(1),data);

    testdata.K=K(U,L);

    prbep_svmp(R)=100-test_prbep(classifier_svmp,testdata.K,Y(U));
    prbep_rlsc(R)=100-test_prbep(classifier_rlsc,testdata.K,Y(U));

    disp(['Laplacian SVM Performance on split ' num2str(R) ': ' num2str(prbep_svmp(R))]);
    disp(['Laplacian RLS Performance on split ' num2str(R) ': ' num2str(prbep_rlsc(R))]);

end
fprintf(1,'\n\n');  
results.err_svm = mean(prbep_svmp);
results.err_rlsc = mean(prbep_rlsc);

disp(['Mean (std dev) Laplacian SVM PRBEP :  ' num2str(mean(prbep_svmp))  '  ( ' num2str(std(prbep_svmp))  ' )']);
disp(['Mean (std dev) Laplacian RLS PRBEP  ' num2str(mean(prbep_rlsc)) '  ( ' num2str(std(prbep_rlsc)) ' )']);

end 

function prbep=test_prbep(classifier,K,Y);
   f=K(:,classifier.svs)*classifier.alpha-classifier.b;
   [m,n,maxcm]=classifier_evaluation(f,Y,'pre_rec_equal');
   prbep=100*(maxcm(1,1)/(maxcm(1,1)+maxcm(1,2)));