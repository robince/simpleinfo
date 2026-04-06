function Iperm = calcinfoperm_slice(X, xb, y, yb, Nperm, bias, beta)
%CALCINFOPERM_SLICE Permutation MI null for each X page against a fixed Y.

if nargin < 6
    bias = true;
end
if nargin < 7
    beta = 0;
end

validateattributes(X, {'numeric'}, {'real', '2d', 'nonempty'});
validateattributes(y, {'numeric'}, {'real', 'vector', 'nonempty'});
y = y(:);
if size(X, 1) ~= numel(y)
    error('calcinfoperm_slice:ShapeMismatch', 'X must have shape [Ntrl, Nx] and Y must have length Ntrl.');
end

nCols = size(X, 2);
Iperm = zeros(Nperm, nCols);
for col = 1:nCols
    Iperm(:, col) = calcinfoperm(X(:, col), xb, y, yb, Nperm, bias, beta);
end
end
