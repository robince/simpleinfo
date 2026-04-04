function I = calccoi(x, xb, y, yb, z, zb, bias, beta)
% I = calccoi(x, xb, y, yb, z, zb)
% calculate co-information between
% discrete data sets x, y, and z
% I = coI( X ; Y ; Z ) = -II( X; Y; Z)
% co-information is negative interaction information
% For co-information, positive values represent net redundancy, 
% negative values represent net synergy
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
    error('calccoi: Number of trials must match')
end
Ntrl = length(x);

% function which calculates the entropy of a probability
% distribution vector
ent = @(p) -sum(p(p(:)>0).*log2(p(p(:)>0)));

% function which calculates the probability histogram from
% a vector of integer trials/samples
counts = accumarray([x+1 y+1 z+1], 1, [xb yb zb]);
Pxyz = (counts+beta)./(Ntrl+beta*numel(counts));

HX = ent(sum(sum(Pxyz,2),3));
HY = ent(sum(sum(Pxyz,1),3));
HZ = ent(sum(sum(Pxyz,1),2));

HXY = ent(sum(Pxyz,3));
HXZ = ent(sum(Pxyz,2));
HYZ = ent(sum(Pxyz,1));

HXYZ = ent(Pxyz);

Inobc = HX + HY + HZ - HXY - HXZ - HYZ + HXYZ;

if bias
    I = Inobc - mmbiascoi(xb, yb, zb, Ntrl);
else
    I = Inobc;
end
