function [I SMI p] = calcsmi(x, xb, y, yb, beta);
% [I SMI p] = calcsmi(x, xb, y, yb)
% calculate samplewise mutual information (SMI) between
% discrete data sets x and y
% x should take values in [0 xb-1]
% y should take values in [0 yb-1]
% beta is add-constant beta probability estimator (0.5 = KT estimator)
% (default 0)
% I = mean(SMI)
% Outputs:
% I - overall MI value I(X;Y)
% SMI - [Nsample 1] vector of SMI value for each sample
% p - p-value for overall MI value

x = x(:);
y = y(:);
if nargin<5
    beta = 0;
end
if length(x) ~= length(y)
    error('calcsmi: Number of trials must match')
end
Ntrl = length(x);
% joint probability distribution
counts = accumarray([x+1 y+1], 1, [xb yb]);
Pxy = (counts+beta)./(Ntrl+beta*numel(counts));

% pointwise mutual information
PMI = zeros(size(Pxy));
idx = Pxy>0;
Pxyind = sum(Pxy,2) * sum(Pxy,1);
PMI(idx) = log2(Pxy(idx)) - log2(Pxyind(idx));

% mutual information
I = sum(Pxy(idx).*PMI(idx));

SMI = PMI(sub2ind(size(PMI),x+1,y+1));

% return p-value if requested
% 2*Ntrl*log(2) * I is chi-square distributed
if nargout>2
    p = 1 - chi2cdf(2*Ntrl*log(2)*I, (xb-1)*(yb-1));
end
