function xbin = eqpop_sorted_reference(xSorted, nb)
xSorted = xSorted(:);
n = numel(xSorted);
if n == 0
    error('fastinfo:eqpop:EmptyInput', 'Input must contain at least one sample.');
end
if nb > n
    error('fastinfo:eqpop:TooManyBins', 'Nb cannot exceed the number of samples.');
end
if any(~isfinite(xSorted))
    error('fastinfo:eqpop:NonFinite', 'Input must contain only finite values.');
end
if any(diff(xSorted) < 0)
    error('fastinfo:eqpop_sorted:NotSorted', 'Input to fastinfo.eqpop_sorted must be sorted.');
end

groupStarts = [1; find(diff(xSorted) ~= 0) + 1; n + 1];
nGroups = numel(groupStarts) - 1;
if nGroups < nb
    error('fastinfo:eqpop:TiedValues', ...
        ['Cannot form the requested number of equal-population bins without ' ...
         'splitting tied values. If the input is discrete or strongly ' ...
         'quantized, use rebin instead.']);
end

prefix = groupStarts - 1;
idealSize = n / nb;
dp = inf(nb + 1, nGroups + 1);
parent = zeros(nb + 1, nGroups + 1);
dp(1, 1) = 0;

for b = 1:nb
    minGroupsUsed = b;
    maxGroupsUsed = nGroups - (nb - b);
    for g = minGroupsUsed:maxGroupsUsed
        bestCost = inf;
        bestParent = NaN;
        for prev = (b - 1):(g - 1)
            prefixCost = dp(b, prev + 1);
            if ~isfinite(prefixCost)
                continue
            end
            count = prefix(g + 1) - prefix(prev + 1);
            deviation = count - idealSize;
            cost = prefixCost + deviation * deviation;
            if cost < bestCost
                bestCost = cost;
                bestParent = prev;
            end
        end
        dp(b + 1, g + 1) = bestCost;
        parent(b + 1, g + 1) = bestParent;
    end
end

if ~isfinite(dp(nb + 1, nGroups + 1))
    error('fastinfo:eqpop:PartitionFailed', ...
        'Failed to construct a tie-consistent equal-population partition.');
end

groupCuts = zeros(nb + 1, 1);
groupCuts(nb + 1) = nGroups;
g = nGroups;
for b = nb:-1:1
    prev = parent(b + 1, g + 1);
    if isnan(prev)
        error('fastinfo:eqpop:PartitionFailed', ...
            'Failed to reconstruct the tie-consistent partition.');
    end
    groupCuts(b) = prev;
    g = prev;
end

xbin = zeros(n, 1, 'int32');
for bin = 0:(nb - 1)
    startIdx = groupStarts(groupCuts(bin + 1) + 1);
    stopIdx = groupStarts(groupCuts(bin + 2) + 1) - 1;
    xbin(startIdx:stopIdx) = int32(bin);
end
end
