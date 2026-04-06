function I = calcinfomatched(X, xb, Y, yb, bias, beta)
%CALCINFOMATCHED Column-wise mutual information for matched X/Y pages.

if nargin < 5
    bias = true;
end
if nargin < 6
    beta = 0;
end

validateattributes(X, {'numeric'}, {'real', '2d', 'nonempty'});
validateattributes(Y, {'numeric'}, {'real', '2d', 'nonempty'});
if ~isequal(size(X), size(Y))
    error('calcinfomatched:ShapeMismatch', 'X and Y must have the same shape.');
end

nCols = size(X, 2);
I = zeros(nCols, 1);
for col = 1:nCols
    I(col) = calcinfo(X(:, col), xb, Y(:, col), yb, bias, beta);
end
end
