function [coI PcoI] = calcpcoi(x, xb, y, yb, z, zb, variant, beta)
% [I p] = calcpco(x, xb, y, yb)
% calculate pointwise co-information between
% discrete data sets x, y and z
% x should take values in [0 xb-1]
% y should take values in [0 yb-1]
% z should take values in [0 zb-1]
% variant is 'weighted', 'none' (default none)
% beta is add-constant beta probability estimator (0.5 = KT estimator)
% (default 0)
% Outputs:
% coI - overall coI value coI(X;Y;Z)
% PcoI - [xb yb zb] array of pointwise co-information values

x = x(:);
y = y(:);
z = z(:);
if nargin<7 || isempty(variant)
    weighted = false;
else
    if strcmpi(variant,'weighted')
        weighted = true;
    else
        weighted = false;
    end
end
if nargin<8
    beta = 0;
end
if length(x) ~= length(y) || length(x) ~= length(z)
    error('calcpcoi: Number of trials must match')
end
Ntrl = length(x);
% joint probability distribution
counts = accumarray([x+1 y+1 z+1],1);
Pxyz = (counts+beta)./(Ntrl+beta*numel(counts));
if size(Pxyz,1)~=xb || size(Pxyz,2)~=yb || size(Pxyz,3)~=zb
    error('calccoi: Problem with data values')
end


Px = sum(sum(Pxyz,2),3);
Py = sum(sum(Pxyz,1),3);
Pz = sum(sum(Pxyz,1),2);

Pxy = sum(Pxyz,3);
Pxz = sum(Pxyz,2);
Pyz = sum(Pxyz,1);

% relies on Matlab automatic singleton expansion since 2016b
PcoI = log2(Pxy) + log2(Pxz) + log2(Pyz) - log2(Px) - log2(Py) - log2(Pz) - log2(Pxyz);

% co-information
idx = Pxyz(:)<eps;
% if Pxyz=0 then that cell does not appear in sum
PcoI(idx)=0;
summand = Pxyz.*PcoI;
coI = sum(summand(:));

if weighted
    PcoI = summand;
end
