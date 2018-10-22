function [I p] = calcinfo(x, xb, y, yb);
% [I p] = calcinfo(x, xb, y, yb)
% calculate mutual information and p value between
% discrete data sets x and y
% I = MI( X ; Y )
% x should take values in [0 xb-1]
% y should take values in [0 yb-1]

x = x(:);
y = y(:);
if length(x) ~= length(y)
    error('calcinfo: Number of trials must match')
end
Ntrl = length(x);

% function which calculates the entropy of a probability
% distribution vector
ent = @(p) -sum(p(p(:)>0).*log2(p(p(:)>0)));

% function which calculates the probability histogram from
% a vector of integer trials/samples 
Pxy = accumarray([x+1 y+1],1)./Ntrl;
if size(Pxy,1) ~= xb || size(Pxy,2) ~= yb
    error('calcinfo: Problem with data values')
end

% mutual information
% I(X;Y) = H(X) + H(Y) - H(X,Y)
% this is the intersection of the entropy between X and Y
% that is the uncertainty that is shared between X and Y
% ie that part of X which can be explained by Y
% or equivalently that part of Y which can be explained by X
Px = sum(Pxy,2);
Py = sum(Pxy,1);
I = ent(Px) + ent(Py) - ent(Pxy);

% return p-value if requested
% 2*Ntrl*log(2) * I is chi-square distributed
% In fact the popular chi-square test was originally developed
% as an approximation to this log-likelihood test (sometimes
% called the G-test) because of the difficulty of calculating
% logarithms before the advent of computers
if nargout>1
    p = 1 - chi2cdf(2*Ntrl*log(2)*I, (xb-1)*(yb-1));
end
