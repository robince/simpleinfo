function [I PMI p] = calcpmi(x, xb, y, yb, variant);
% [I p] = calcpmi(x, xb, y, yb)
% calculate mutual information and p value between
% discrete data sets x and y
% I = MI( X ; Y )
% x should take values in [0 xb-1]
% y should take values in [0 yb-1]
% variant is 'normalised', 'weighted', 'none'

x = x(:);
y = y(:);
if nargin<5
    weighted = false;
end
if length(x) ~= length(y)
    error('calcinfo: Number of trials must match')
end
Ntrl = length(x);
% joint probability distribution
Pxy = accumarray([x+1 y+1],1)./Ntrl;
if size(Pxy,1) ~= xb || size(Pxy,2) ~= yb
    error('calcinfo: Problem with data values')
end

% pointwise mutual information
PMI = log2(Pxy) - bsxfun(@plus,log2(sum(Pxy,1)),log2(sum(Pxy,2)));

if weighted
    PMI = Pxy.*PMI;
end

% mutual information
I = sum(sum(Pxy.*PMI));

% return p-value if requested
% 2*Ntrl*log(2) * I is chi-square distributed
if nargout>2
    p = 1 - chi2cdf(2*Ntrl*log(2)*I, (xb-1)*(yb-1));
end
