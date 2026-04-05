function pairI = calcpairwiseinfo_slice(X, xb, y, yb, varargin)
p = inputParser;
p.addParameter('Bias', false, @(v) islogical(v) && isscalar(v));
p.addParameter('Validate', true, @(v) islogical(v) && isscalar(v));
p.parse(varargin{:});
opts = p.Results;

if opts.Validate
    validateattributes(X, {'numeric'}, {'real', '2d', 'nonempty'});
    validateattributes(y, {'numeric'}, {'real', 'vector', 'nonempty'});
end

nPairs = nchoosek(yb, 2);
pairI = zeros(nPairs, size(X, 2));
for col = 1:size(X, 2)
    pairI(:, col) = fastinfo.calcpairwiseinfo(X(:, col), xb, y, yb, 'Bias', opts.Bias, 'Validate', false);
end
end
