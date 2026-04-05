function pairI = calcpairwiseinfo(X, xb, y, yb, varargin)
p = inputParser;
p.addParameter('Bias', false, @(v) islogical(v) && isscalar(v));
p.addParameter('Validate', true, @(v) islogical(v) && isscalar(v));
p.parse(varargin{:});
opts = p.Results;

if opts.Validate
    validateattributes(X, {'numeric'}, {'real', 'vector', 'nonempty'});
    validateattributes(y, {'numeric'}, {'real', 'vector', 'nonempty'});
end

[Xs, idx] = sort(X(:), 'ascend');
Ys = y(:);
Ys = Ys(idx);
nPairs = nchoosek(yb, 2);
pairI = zeros(nPairs, 1);
pairIndex = 1;
for yi = 0:(yb - 2)
    iidx = (Ys == yi);
    for yj = (yi + 1):(yb - 1)
        pidx = iidx | (Ys == yj);
        px = Xs(pidx);
        py = Ys(pidx);
        py(py == yi) = 0;
        py(py == yj) = 1;
        qpx = fastinfo.eqpop_sorted(px, xb, 'WarnOnTies', false);
        pairI(pairIndex) = fastinfo.calcinfo(qpx, xb, py, 2, 'Bias', opts.Bias);
        pairIndex = pairIndex + 1;
    end
end
end
