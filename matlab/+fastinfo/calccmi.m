function I = calccmi(x, xb, y, yb, z, zb, varargin)
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
    validateattributes(z, {'numeric'}, {'real', 'nonempty'});
    validateattributes(xb, {'numeric'}, {'real', 'scalar', 'integer', 'positive'});
    validateattributes(yb, {'numeric'}, {'real', 'scalar', 'integer', 'positive'});
    validateattributes(zb, {'numeric'}, {'real', 'scalar', 'integer', 'positive'});
end

if exist('fastinfo_calccmi_cpp', 'file') == 3
    validate_native_integer_class(x, 'x');
    validate_native_integer_class(y, 'y');
    validate_native_integer_class(z, 'z');
    I = fastinfo_calccmi_cpp(x, double(xb), y, double(yb), z, double(zb));
else
    I = feval('calccmi', x, xb, y, yb, z, zb, false, 0);
end

if opts.Bias
    I = I - mmbiascmi(xb, yb, zb, numel(x));
end
end
