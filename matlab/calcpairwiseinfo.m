function pairI = calcpairwiseinfo(X, xb, y, yb, bias)
%CALCPAIRWISEINFO Pairwise binary MI after pair-specific equipop binning.

if nargin < 5
    bias = false;
end

validateattributes(X, {'numeric'}, {'real', 'vector', 'nonempty'});
validateattributes(y, {'numeric'}, {'real', 'vector', 'nonempty'});
X = X(:);
y = y(:);
if numel(X) ~= numel(y)
    error('calcpairwiseinfo:ShapeMismatch', 'X and Y must have the same number of samples.');
end

[Xs, idx] = sort(X);
Ys = y(idx);
pairI = zeros(nchoosek(yb, 2), 1);
pairIndex = 1;
for yi = 0:(yb - 2)
    iidx = Ys == yi;
    for yj = (yi + 1):(yb - 1)
        pidx = iidx | (Ys == yj);
        px = Xs(pidx);
        py = Ys(pidx);
        py(py == yi) = 0;
        py(py == yj) = 1;
        qpx = iEqpopSortedReference(px, xb);
        pairI(pairIndex) = calcinfo(qpx, xb, py, 2, bias, 0);
        pairIndex = pairIndex + 1;
    end
end
end

function labels = iEqpopSortedReference(xSorted, nb)
n = numel(xSorted);
groupStarts = [1; find(diff(xSorted) ~= 0) + 1; n + 1];
nGroups = numel(groupStarts) - 1;
if nGroups < nb
    error('calcpairwiseinfo:Ties', ...
        'Cannot form the requested number of equal-population bins without splitting tied values.');
end

ideal = n / nb;
dp = inf(nb + 1, nGroups + 1);
parent = zeros(nb + 1, nGroups + 1);
dp(1, 1) = 0;
for b = 1:nb
    minUsed = b;
    maxUsed = nGroups - (nb - b);
    for g = minUsed:maxUsed
        for prev = (b - 1):(g - 1)
            prefix = dp(b, prev + 1);
            if ~isfinite(prefix)
                continue
            end
            count = groupStarts(g + 1) - groupStarts(prev + 1);
            deviation = count - ideal;
            cost = prefix + deviation.^2;
            if cost < dp(b + 1, g + 1)
                dp(b + 1, g + 1) = cost;
                parent(b + 1, g + 1) = prev;
            end
        end
    end
end

cuts = zeros(nb + 1, 1);
cuts(end) = nGroups;
g = nGroups;
for b = nb:-1:1
    prev = parent(b + 1, g + 1);
    cuts(b) = prev;
    g = prev;
end

labels = zeros(n, 1);
for b = 1:nb
    startIdx = groupStarts(cuts(b) + 1);
    stopIdx = groupStarts(cuts(b + 1) + 1) - 1;
    labels(startIdx:stopIdx) = b - 1;
end
end
