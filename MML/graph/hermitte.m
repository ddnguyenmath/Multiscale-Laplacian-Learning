function [ w ] = hermitte( x, n )
%HERMITTE Summary of this function goes here
%   Detailed explanation goes here
w0 = pi^(-1/4);
w1 = sqrt(2)*w0*x;
if n == 0
    w = w0;
elseif n == 1
    w = w1;
else
    for i=2:n
        w2 = x*w1 - sqrt((i-1)/2)*w0;
        w2 = w2/sqrt(i/2);
        w0 = w1;
        w1 = w2;
    end
    w = w2;
end
        
end

