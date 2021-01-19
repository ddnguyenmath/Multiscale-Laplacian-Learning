function results=mlap(dataset,options,M) 
% dataset.mat should have matrices 
% X (n x d training data matrix, n examples, d features)
% Xt (test data matrix) (if available)     
% training labels Y      
% test labels Yt (if available)
% Coded  originally downloaded from
%   Sindhwani, Vikas, Partha Niyogi, and Mikhail Belkin. 
%    "Beyond the point cloud: from transductive to semi-supervised learning." 
%      Proceedings of the 22nd international conference on Machine learning. 2005.

setpaths

 load(dataset);
C=unique(Y);
if length(C)==2
             C=[1 -1]; nclassifiers=1;
else
               nclassifiers=length(C);
end



%if ~exist('M','var') 
hermite_order = size(options,2)
for i=1:hermite_order
  M = full(laplacian_hermite(options(i),X));
  M=M^options(i).LaplacianDegree;
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
if exist('folds','var')
   number_runs=size(folds,2);
else
   number_runs=size(idxLabs,1); 
end
for R=1:number_runs
	if exist('folds','var')
      L=find(folds(:,R));
    else
      L=idxLabs(R,:);
    end
    U=(1:size(K,1))'; 
    U(L)=[];
	  data.K=K(L,L); data.X=X(L,:); 
 
    fsvm=[];
    frlsc=[];
    fsvm_t=[];
    frlsc_t=[];
 
    for c=1:nclassifiers
        if nclassifiers>1
            %fprintf(1,'Class %d versus rest\n',C(c)); 
        end
        data.Y=(Y(L)==C(c))-(Y(L)~=C(c)); % labeled data
        classifier_svm=svmp(options(1),data);
        classifier_rlsc=rlsc(options(1),data);
         
        fsvm(:,c)=K(U,L(classifier_svm.svs))*classifier_svm.alpha-classifier_svm.b;
        frlsc(:,c)=K(U,L(classifier_rlsc.svs))*classifier_rlsc.alpha-classifier_rlsc.b;

     
        if exist('bias','var')
          [fsvm(:,c),classifier_svm.b]  = adjustbias(fsvm(:,c)+classifier_svm.b,  bias);
          [frlsc(:,c),classifier_rlsc.b] = adjustbias(frlsc(:,c)+classifier_rlsc.b,bias);
        end
 
        results(R).fsvm(:,c)=fsvm(:,c);
        results(R).frlsc(:,c)=frlsc(:,c); 
        yu=(Y(U)==C(c))-(Y(U)~=C(c));
 
        if exist('Xt','var')
            fsvm_t(:,c)=Kt(:,L(classifier_svm.svs))*classifier_svm.alpha-classifier_svm.b;
            frlsc_t(:,c)=Kt(:,L(classifier_rlsc.svs))*classifier_rlsc.alpha-classifier_rlsc.b;
            results(R).fsvm_t(:,c)=fsvm_t(:,c);
            results(R).frlsc_t(:,c)=frlsc_t(:,c);
            yt=(Yt==C(c))-(Yt~=C(c));
        end
    end
  
 
   
   if nclassifiers==1
        fsvm=sign(results(R).fsvm);
        frlsc=sign(results(R).frlsc);
        if exist('Xt','var')
          fsvm_t=sign(results(R).fsvm_t);
          frlsc_t=sign(results(R).frlsc_t);
        end
   else
        [e,fsvm]=max(results(R).fsvm,[],2); fsvm=C(fsvm);
        [e,frlsc]=max(results(R).frlsc,[],2); frlsc=C(frlsc);
        if exist('Xt','var')
          [e,fsvm_t]=max(results(R).fsvm_t,[],2); fsvm_t=C(fsvm_t);
          [e,frlsc_t]=max(results(R).frlsc_t,[],2); frlsc_t=C(frlsc_t);
        end
   end

   cm=confusion(fsvm,Y(U)); results(R).cm_svm=cm; 
   results(R).err_svm=100*(1-sum(diag(cm))/sum(cm(:)));
   cm=confusion(frlsc,Y(U)); results(R).cm_rlsc=cm; 
   results(R).err_rlsc=100*(1-sum(diag(cm))/sum(cm(:)));
        
   if exist('Xt','var')
        cm=confusion(fsvm_t,Yt); results(R).cm_svm_t=cm; 
        results(R).err_svm_t=100*(1-sum(diag(cm))/sum(cm(:)));
        cm=confusion(frlsc_t,Yt); results(R).cm_rlsc_t=cm; 
        results(R).err_rlsc_t=100*(1-sum(diag(cm))/sum(cm(:)));
   end
        
fprintf(1,'split=%d LapSVM (transduction) err = %f \n',R, results(R).err_svm);
fprintf(1,'split=%d LapRLS (transduction) err = %f \n',R, results(R).err_rlsc);

     
if exist('Xt','var')
    fprintf(1,'split=%d LapSVM (out-of-sample) err = %f\n',R, results(R).err_svm_t);
    fprintf(1,'split=%d LapRLS (out-of-sample) err = %f\n',R, results(R).err_rlsc_t);
end


end



fprintf(1,'\n\n');
% disp('LapSVM (transduction) mean confusion matrix');
% disp(round(mean(reshape(vertcat(results.cm_svm)',[size(results(1).cm_svm,1) size(results(1).cm_svm,1) length(results)]),3)));
% disp('LapRLS (transduction) mean confusion matrix');
% disp(round(mean(reshape(vertcat(results.cm_rlsc)',[size(results(1).cm_rlsc,1) size(results(1).cm_rlsc,1) length(results)]),3)));
fprintf(1,'LapSVM (transduction mean(std)) err = %f (%f) \n',mean(vertcat(results.err_svm)),std(vertcat(results.err_svm)));  
fprintf(1,'LapRLS (transduction mean(std)) err = %f (%f)  \n',mean(vertcat(results.err_rlsc)),std(vertcat(results.err_rlsc)));
if exist('Xt','var')
    fprintf(1,'\n\n');
    disp('LapSVM (out-of-sample) mean confusion matrix');
    disp(round(mean(reshape(vertcat(results.cm_svm_t)',[size(results(1).cm_svm_t,1) size(results(1).cm_svm_t,1) length(results)]),3)));
    disp('LapRLS (out-of-sample) mean confusion matrix');
    disp(round(mean(reshape(vertcat(results.cm_rlsc_t)',[size(results(1).cm_rlsc_t,1) size(results(1).cm_rlsc_t,1) length(results)]),3)));
    fprintf(1,'LapSVM (out-of-sample mean(std)) err = %f (%f)\n',mean(vertcat(results.err_svm_t)), std(vertcat(results.err_svm_t)));  
    fprintf(1,'LapRLS (out-of-sample mean(std)) err = %f (%f)\n',mean(vertcat(results.err_rlsc_t)),std(vertcat(results.err_rlsc_t)));
end


  
function [f1,b]=adjustbias(f,bias)
     jj=ceil((1-bias)*length(f));
     if jj >= length(f)
         jj = length(f)-1;
     end
     g=sort(f);
     b=0.5*(g(jj)+g(jj+1));
     f1=f-b;
