function [I, IK] = calccondcmi(x, xb, y, yb, z, zb, k, kb)
%CALCCONDMI Conditional MI plus weighted per-K contributions.

validateattributes(x, {'numeric'}, {'real', 'vector', 'nonempty'});
validateattributes(y, {'numeric'}, {'real', 'vector', 'nonempty'});
validateattributes(z, {'numeric'}, {'real', 'vector', 'nonempty'});
validateattributes(k, {'numeric'}, {'real', 'vector', 'nonempty'});

x = x(:);
y = y(:);
z = z(:);
k = k(:);
if numel(x) ~= numel(y) || numel(x) ~= numel(z) || numel(x) ~= numel(k)
    error('calccondcmi:ShapeMismatch', 'X, Y, Z, and K must have the same number of samples.');
end

Ntrl = numel(x);
ent = @(p) -sum(p(p(:) > 0) .* log2(p(p(:) > 0)));

counts = accumarray([x + 1, y + 1, z + 1], 1, [xb, yb, zb]);
Pxyz = counts ./ Ntrl;
I = ent(sum(Pxyz, 2)) + ent(sum(Pxyz, 1)) - ent(Pxyz) - ent(sum(sum(Pxyz, 1), 2));

IK = zeros(kb, 1);
for ki = 0:(kb - 1)
    idx = (k == ki);
    if ~any(idx)
        IK(ki + 1) = 0;
        continue
    end
    countsK = accumarray([x(idx) + 1, y(idx) + 1, z(idx) + 1], 1, [xb, yb, zb]);
    PxyzK = countsK ./ Ntrl;
    IK(ki + 1) = ent(sum(PxyzK, 2)) + ent(sum(PxyzK, 1)) - ent(PxyzK) - ent(sum(sum(PxyzK, 1), 2));
end
end
