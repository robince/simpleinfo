function [I SMI p] = calcsmi(x, xb, y, yb);
% [I SMI p] = calcsmi(x, xb, y, yb)
% calculate sample mutual information and p value between
% discrete data sets x and y
% I = MI( X ; Y )
% x should take values in [0 xb-1]
% y should take values in [0 yb-1]
% SMI is pointwise MI evaluate at each input sample
% I = mean(SMI)

x = x(:);
y = y(:);
if nargin<5
    weighted = false;
end
if length(x) ~= length(y)
    error('calcsmi: Number of trials must match')
end
Ntrl = length(x);
% joint probability distribution
Pxy = accumarray([x+1 y+1],1)./Ntrl;
if size(Pxy,1) ~= xb || size(Pxy,2) ~= yb
    error('calcsmi: Problem with data values')
end

% pointwise mutual information
PMI = log2(Pxy) - bsxfun(@plus,log2(sum(Pxy,1)),log2(sum(Pxy,2)));

% mutual information
idx = Pxy(:)>0;
I = sum(Pxy(idx).*PMI(idx));

SMI = PMI(sub2ind(size(PMI),x+1,y+1));

% return p-value if requested
% 2*Ntrl*log(2) * I is chi-square distributed
if nargout>2
    p = 1 - chi2cdf(2*Ntrl*log(2)*I, (xb-1)*(yb-1));
end
