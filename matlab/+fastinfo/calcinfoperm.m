function Iperm = calcinfoperm(x, xb, y, yb, nperm, varargin)
p = inputParser;
p.addParameter('Threads', [], @(v) isempty(v) || (isscalar(v) && isnumeric(v)));
p.addParameter('Validate', true, @(v) islogical(v) && isscalar(v));
p.addParameter('Bias', false, @(v) islogical(v) && isscalar(v));
p.addParameter('Seed', 5489, @(v) isnumeric(v) && isscalar(v) && isfinite(v) && v >= 0 && floor(v) == v);
p.parse(varargin{:});
opts = p.Results;

ensure_native_path();

if opts.Validate
    validateattributes(x, {'numeric'}, {'real', 'nonempty'});
    validateattributes(y, {'numeric'}, {'real', 'nonempty'});
    validateattributes(xb, {'numeric'}, {'real', 'scalar', 'integer', 'positive'});
    validateattributes(yb, {'numeric'}, {'real', 'scalar', 'integer', 'positive'});
    validateattributes(nperm, {'numeric'}, {'real', 'scalar', 'integer', 'nonnegative'});
end

nThreads = resolve_threads(opts.Threads);
if exist('fastinfo_calcinfoperm_cpp', 'file') == 3
    validate_native_integer_class(x, 'x');
    validate_native_integer_class(y, 'y');
    Iperm = fastinfo_calcinfoperm_cpp(x, double(xb), y, double(yb), double(nperm), double(nThreads), double(opts.Seed));
else
    rng(opts.Seed, 'twister');
    Iperm = feval('calcinfoperm', x, xb, y, yb, nperm, false, 0);
end

if opts.Bias
    Iperm = Iperm - mmbiasinfo(xb, yb, numel(x));
end
end
