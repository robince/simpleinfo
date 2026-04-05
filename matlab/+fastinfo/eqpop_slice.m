function xbin = eqpop_slice(X, nb, varargin)
p = inputParser;
p.addParameter('Threads', [], @(v) isempty(v) || (isscalar(v) && isnumeric(v)));
p.addParameter('Validate', true, @(v) islogical(v) && isscalar(v));
p.addParameter('WarnOnTies', true, @(v) islogical(v) && isscalar(v));
p.parse(varargin{:});
opts = p.Results;

ensure_native_path();

if opts.Validate
    validateattributes(X, {'double'}, {'real', '2d', 'nonempty'});
    validateattributes(nb, {'numeric'}, {'real', 'scalar', 'integer', 'positive'});
end
nThreads = resolve_threads(opts.Threads);

if exist('fastinfo_eqpop_slice_cpp', 'file') == 3
    xbin = fastinfo_eqpop_slice_cpp(double(X), double(nb), double(nThreads));
else
    xbin = nan(size(X));
    for col = 1:size(X, 2)
        try
            xbin(:, col) = double(eqpop_reference(double(X(:, col)), double(nb)));
        catch
            xbin(:, col) = nan(size(X, 1), 1);
        end
    end
end

maybe_warn_failed_pages(xbin, opts.WarnOnTies, 'fastinfo.eqpop_slice');
end
