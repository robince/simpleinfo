function I = calcinfo_slice(X, xb, y, yb, varargin)
p = inputParser;
p.addParameter('Threads', [], @(v) isempty(v) || (isscalar(v) && isnumeric(v)));
p.addParameter('Validate', true, @(v) islogical(v) && isscalar(v));
p.addParameter('Bias', false, @(v) islogical(v) && isscalar(v));
p.parse(varargin{:});
opts = p.Results;

ensure_native_path();

if opts.Validate
    validateattributes(X, {'numeric'}, {'real', '2d', 'nonempty'});
    validateattributes(y, {'numeric'}, {'real', 'vector', 'numel', size(X, 1)});
    validateattributes(xb, {'numeric'}, {'real', 'scalar', 'integer', 'positive'});
    validateattributes(yb, {'numeric'}, {'real', 'scalar', 'integer', 'positive'});
end

nThreads = resolve_threads(opts.Threads);
if exist('fastinfo_calcinfo_slice_cpp', 'file') == 3
    I = fastinfo_calcinfo_slice_cpp(X, double(xb), y, double(yb), double(nThreads));
else
    I = zeros(size(X, 2), 1);
    for col = 1:size(X, 2)
        I(col) = feval('calcinfo', X(:, col), xb, y, yb, false, 0);
    end
end

if opts.Bias
    I = I - mmbiasinfo(xb, yb, size(X, 1));
end
end
