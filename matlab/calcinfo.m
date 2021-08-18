function [I, p, Pxy] = calcinfo(x, xb, y, yb, bias, beta)
% [I p] = calcinfo(x, xb, y, yb, beta)
% calculate mutual information and p value between
% discrete data sets x and y
% I = MI( X ; Y )
% x should take values in [0 xb-1]
% y should take values in [0 yb-1]
% beta is add-constant beta probability estimator (0.5 = KT estimator)
% (default 0)
% Outputs:
% I - MI  I(X;Y)
% p - p-value for MI (analytic from chi-square distribution)
% beta - add-constant beta probability estimator (0.5 = KT estimator)
% (default 0)

if nargin<5
    bias = true;
end
if nargin<6
    beta = 0;
end

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
counts = (accumarray([x+1 y+1],1)+beta);
Pxy = (counts+beta)./(Ntrl+beta*numel(counts));
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
Inobc = ent(Px) + ent(Py) - ent(Pxy);

% apply simple subtactive bias correction if required
% (subtracting the mean of the analytic chi-squared null distribution)
if bias
    I = Inobc - mmbiasinfo(xb,yb,Ntrl);
else
    I = Inobc;
end

% return p-value if requested
% 2*Ntrl*log(2) * I is chi-square distributed
% In fact the popular chi-square test was originally developed
% as an approximation to this log-likelihood test (sometimes
% called the G-test) because of the difficulty of calculating
% logarithms before the advent of computers
if nargout>1
    p = 1 - chi2cdf(2*Ntrl*log(2)*Inobc, (xb-1)*(yb-1));
end
