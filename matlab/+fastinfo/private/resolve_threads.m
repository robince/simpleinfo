function nThreads = resolve_threads(nThreads)
if nargin < 1 || isempty(nThreads)
    try
        nThreads = maxNumCompThreads; %#ok<MXMNC>
    catch
        nThreads = 1;
    end
end
nThreads = max(1, double(nThreads));
if ~isscalar(nThreads) || ~isfinite(nThreads) || floor(nThreads) ~= nThreads
    error('fastinfo:Threads', 'Threads must be a positive integer scalar.');
end
end
