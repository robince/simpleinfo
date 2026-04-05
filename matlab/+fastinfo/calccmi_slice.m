function I = calccmi_slice(X, xb, y, yb, z, zb, varargin)
p = inputParser;
p.addParameter('Threads', [], @(v) isempty(v) || (isscalar(v) && isnumeric(v)));
p.addParameter('Validate', true, @(v) islogical(v) && isscalar(v));
p.addParameter('Bias', false, @(v) islogical(v) && isscalar(v));
p.parse(varargin{:});
opts = p.Results;

ensure_native_path();
if opts.Validate
    validateattributes(X, {'numeric'}, {'real', '2d', 'nonempty'});
    validateattributes(y, {'numeric'}, {'real', 'vector', 'nonempty'});
    validateattributes(z, {'numeric'}, {'real', 'vector', 'nonempty'});
end

nThreads = resolve_threads(opts.Threads);
if exist('fastinfo_calccmi_slice_cpp', 'file') == 3
    I = fastinfo_calccmi_slice_cpp(X, double(xb), y, double(yb), z, double(zb), double(nThreads));
else
    I = feval('calccmi_slice', X, xb, y, yb, z, zb, false, 0);
end

if opts.Bias
    I = I - mmbiascmi(xb, yb, zb, size(X, 1));
end
end
