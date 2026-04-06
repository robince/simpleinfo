function xbin = eqpop_sorted(xSorted, nb, varargin)
p = inputParser;
p.addParameter('Validate', true, @(v) islogical(v) && isscalar(v));
p.addParameter('WarnOnTies', true, @(v) islogical(v) && isscalar(v));
p.parse(varargin{:});
opts = p.Results;

ensure_native_path();

if opts.Validate
    validateattributes(xSorted, {'double'}, {'real', 'vector', 'nonempty'});
    validateattributes(nb, {'numeric'}, {'real', 'scalar', 'integer', 'positive'});
    if any(diff(xSorted(:)) < 0)
        error('fastinfo:eqpop_sorted:NotSorted', ...
            'Input to fastinfo.eqpop_sorted must be sorted in nondecreasing order.');
    end
end
warn_if_quantized(xSorted, nb, opts.WarnOnTies, 'fastinfo.eqpop_sorted');

if exist('fastinfo_eqpop_sorted_cpp', 'file') == 3
    xbin = fastinfo_eqpop_sorted_cpp(double(xSorted), double(nb));
else
    xShape = size(xSorted);
    xbin = reshape(eqpop_sorted_reference(double(xSorted(:)), double(nb)), xShape);
end
end
