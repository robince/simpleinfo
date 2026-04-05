function xbin = eqpop(x, nb, varargin)
p = inputParser;
p.addParameter('Validate', true, @(v) islogical(v) && isscalar(v));
p.addParameter('WarnOnTies', true, @(v) islogical(v) && isscalar(v));
p.parse(varargin{:});
opts = p.Results;

ensure_native_path();

if opts.Validate
    validateattributes(x, {'double'}, {'real', 'vector', 'nonempty'});
    validateattributes(nb, {'numeric'}, {'real', 'scalar', 'integer', 'positive'});
end
warn_if_quantized(x, nb, opts.WarnOnTies, 'fastinfo.eqpop');

if exist('fastinfo_eqpop_cpp', 'file') == 3
    xbin = fastinfo_eqpop_cpp(double(x), double(nb));
else
    xbin = eqpop_reference(double(x), double(nb));
end
end
