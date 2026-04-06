function pairI = calcpairwiseinfo_slice(X, xb, y, yb, bias)
%CALCPAIRWISEINFO_SLICE Pairwise binary MI for each X page.

if nargin < 5
    bias = false;
end

validateattributes(X, {'numeric'}, {'real', '2d', 'nonempty'});
validateattributes(y, {'numeric'}, {'real', 'vector', 'nonempty'});
y = y(:);
if size(X, 1) ~= numel(y)
    error('calcpairwiseinfo_slice:ShapeMismatch', 'X must have shape [Ntrl, Nx] and Y must have length Ntrl.');
end

nPairs = nchoosek(yb, 2);
pairI = zeros(nPairs, size(X, 2));
for col = 1:size(X, 2)
    pairI(:, col) = calcpairwiseinfo(X(:, col), xb, y, yb, bias);
end
end
