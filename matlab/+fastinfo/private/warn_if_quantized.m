function warn_if_quantized(x, nb, warnOnTies, funcName)
if ~warnOnTies
    return
end
x = x(:);
if isempty(x)
    return
end
nUnique = numel(unique(x));
if nUnique < numel(x) && nUnique <= 2 * nb
    warning('fastinfo:QuantizedInput', ...
        '%s received heavily repeated values (%d unique values for %d requested bins). If the input is effectively discrete, use rebin instead.', ...
        funcName, nUnique, nb);
end
end
