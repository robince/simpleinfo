function xbin = eqpop_sorted_slice(Xsorted, nb, varargin)
p = inputParser;
p.addParameter('Threads', [], @(v) isempty(v) || (isscalar(v) && isnumeric(v)));
p.addParameter('Validate', true, @(v) islogical(v) && isscalar(v));
p.addParameter('WarnOnTies', true, @(v) islogical(v) && isscalar(v));
p.parse(varargin{:});
opts = p.Results;

ensure_native_path();

if opts.Validate
    validateattributes(Xsorted, {'double'}, {'real', '2d', 'nonempty'});
    validateattributes(nb, {'numeric'}, {'real', 'scalar', 'integer', 'positive'});
end
nThreads = resolve_threads(opts.Threads);

if exist('fastinfo_eqpop_sorted_slice_cpp', 'file') == 3
    xbin = fastinfo_eqpop_sorted_slice_cpp(double(Xsorted), double(nb), double(nThreads));
else
    xbin = nan(size(Xsorted));
    for col = 1:size(Xsorted, 2)
        try
            xbin(:, col) = double(eqpop_sorted_reference(double(Xsorted(:, col)), double(nb)));
        catch
            xbin(:, col) = nan(size(Xsorted, 1), 1);
        end
    end
end

maybe_warn_failed_pages(xbin, opts.WarnOnTies, 'fastinfo.eqpop_sorted_slice');
end
