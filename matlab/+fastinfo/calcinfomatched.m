function I = calcinfomatched(X, xb, Y, yb, varargin)
p = inputParser;
p.addParameter('Threads', [], @(v) isempty(v) || (isscalar(v) && isnumeric(v)));
p.addParameter('Validate', true, @(v) islogical(v) && isscalar(v));
p.addParameter('Bias', false, @(v) islogical(v) && isscalar(v));
p.parse(varargin{:});
opts = p.Results;

ensure_native_path();
if opts.Validate
    validateattributes(X, {'numeric'}, {'real', '2d', 'nonempty'});
    validateattributes(Y, {'numeric'}, {'real', '2d', 'nonempty'});
    if ~isequal(size(X), size(Y))
        error('fastinfo:calcinfomatched:ShapeMismatch', 'X and Y must have the same shape.');
    end
end

nThreads = resolve_threads(opts.Threads);
if exist('fastinfo_calcinfomatched_cpp', 'file') == 3
    I = fastinfo_calcinfomatched_cpp(X, double(xb), Y, double(yb), double(nThreads));
else
    I = feval('calcinfomatched', X, xb, Y, yb, false, 0);
end

if opts.Bias
    I = I - mmbiasinfo(xb, yb, size(X, 1));
end
end
