function xbin = eqpop_reference(x, nb)
xShape = size(x);
xFlat = x(:);
[xSorted, order] = sort(xFlat, 'ascend');
xSortedBin = eqpop_sorted_reference(xSorted, nb);
xFlatBin = zeros(size(xSortedBin), 'like', xSortedBin);
xFlatBin(order) = xSortedBin;
xbin = reshape(xFlatBin, xShape);
end
