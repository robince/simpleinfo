function [I p] = calccmi(x, xb, y, yb, z, zb);
% [I p] = calccmi(x, xb, y, yb, z, zb)
% calculate conditional mutual information and p value between
% discrete data sets x and y, conditioning out z
% I = MI( X ; Y | Z )
% x should take values in [0 xb-1]
% y should take values in [0 yb-1]
% z should take values in [0 zb-1]

x = x(:);
y = y(:);
z = z(:);
if length(x)~=length(y) || length(x)~=length(z)
    error('calccmi: Number of trials must match')
end
Ntrl = length(x);

% function which calculates the entropy of a probability
% distribution vector
ent = @(p) -sum(p(p(:)>0).*log2(p(p(:)>0)));

% function which calculates the probability histogram from
% a vector of integer trials/samples 
Pxyz = accumarray([x+1 y+1 z+1],1)./Ntrl;
if size(Pxyz,1)~=xb || size(Pxyz,2)~=yb || size(Pxyz,3)~=zb
    error('calcinfo: Problem with data values')
end

% conditional mutual information
% I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
HXZ = ent(sum(Pxyz,2));
HYZ = ent(sum(Pxyz,1));
HXYZ = ent(Pxyz);
HZ = ent(sum(sum(Pxyz,1),2));
I = HXZ + HYZ - HXYZ - HZ;


% return p-value if requested
% 2*Ntrl*log(2) * I is chi-square distributed
% In fact the popular chi-square test was originally developed
% as an approximation to this log-likelihood test (sometimes
% called the G-test) because of the difficulty of calculating
% logarithms before the advent of computers
if nargout>1
    p = 1 - chi2cdf(2*Ntrl*log(2)*I, zb*(xb-1)*(yb-1));
end
