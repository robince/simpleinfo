function I = calccmi_slice(X, xb, y, yb, z, zb, bias, beta)
%CALCCMI_SLICE Column-wise conditional mutual information for matched X pages.

if nargin < 7
    bias = true;
end
if nargin < 8
    beta = 0;
end

validateattributes(X, {'numeric'}, {'real', '2d', 'nonempty'});
validateattributes(y, {'numeric'}, {'real', 'vector', 'nonempty'});
validateattributes(z, {'numeric'}, {'real', 'vector', 'nonempty'});

y = y(:);
z = z(:);
if size(X, 1) ~= numel(y) || size(X, 1) ~= numel(z)
    error('calccmi_slice:ShapeMismatch', 'X must have shape [Ntrl, Nx], Y and Z must have length Ntrl.');
end

nCols = size(X, 2);
I = zeros(nCols, 1);
for col = 1:nCols
    I(col) = calccmi(X(:, col), xb, y, yb, z, zb, bias, beta);
end
end
