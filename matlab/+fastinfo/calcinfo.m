function I = calcinfo(x, xb, y, yb, varargin)
p = inputParser;
p.addParameter('Threads', [], @(v) isempty(v) || (isscalar(v) && isnumeric(v)));
p.addParameter('Validate', true, @(v) islogical(v) && isscalar(v));
p.addParameter('Bias', false, @(v) islogical(v) && isscalar(v));
p.parse(varargin{:});
opts = p.Results;

ensure_native_path();

if opts.Validate
    validateattributes(x, {'numeric'}, {'real', 'nonempty'});
    validateattributes(y, {'numeric'}, {'real', 'nonempty'});
    validateattributes(xb, {'numeric'}, {'real', 'scalar', 'integer', 'positive'});
    validateattributes(yb, {'numeric'}, {'real', 'scalar', 'integer', 'positive'});
end

if exist('fastinfo_calcinfo_cpp', 'file') == 3
    I = fastinfo_calcinfo_cpp(x, double(xb), y, double(yb));
else
    I = feval('calcinfo', x, xb, y, yb, false, 0);
end

if opts.Bias
    I = I - mmbiasinfo(xb, yb, numel(x));
end
end
