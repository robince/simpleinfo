function [I p] = calccmi(x, xb, y, yb, z, zb, bias, beta);
% [I p] = calccmi(x, xb, y, yb, z, zb)
% calculate conditional mutual information and p value between
% discrete data sets x and y, conditioning out z
% I = MI( X ; Y | Z )
% x should take values in [0 xb-1]
% y should take values in [0 yb-1]
% z should take values in [0 zb-1]

if nargin<7
    bias = true;
end
if nargin<8
    beta = 0.;
end

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
counts = accumarray([x+1 y+1 z+1],1);
Pxyz = (counts+beta)./(Ntrl+beta*numel(counts));
if size(Pxyz,1)~=xb || size(Pxyz,2)~=yb || size(Pxyz,3)~=zb
    error('calcinfo: Problem with data values')
end

% conditional mutual information
% I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
HXZ = ent(sum(Pxyz,2));
HYZ = ent(sum(Pxyz,1));
HXYZ = ent(Pxyz);
HZ = ent(sum(sum(Pxyz,1),2));
Inobc = HXZ + HYZ - HXYZ - HZ;

if bias
    I = Inobc - mmbiascmi(xb, yb, zb, Ntrl);
else
    I = Inobc
end

% return p-value if requested
% 2*Ntrl*log(2) * I is chi-square distributed
% In fact the popular chi-square test was originally developed
% as an approximation to this log-likelihood test (sometimes
% called the G-test) because of the difficulty of calculating
% logarithms before the advent of computers
if nargout>1
    p = 1 - chi2cdf(2*Ntrl*log(2)*Inobc, zb*(xb-1)*(yb-1));
end
