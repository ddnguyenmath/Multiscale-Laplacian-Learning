function [f1,b]=adjustbias(f,bias)
     jj=ceil((1-bias)*length(f));
     g=sort(f);
     b=0.5*(g(jj)+g(jj+1));
     f1=f-b;
     