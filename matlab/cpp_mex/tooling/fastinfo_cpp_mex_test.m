function results = fastinfo_cpp_mex_test(varargin)
p = inputParser;
p.addParameter('Compile', false, @(x) islogical(x) && isscalar(x));
p.parse(varargin{:});
opts = p.Results;

cfg = fastinfo_cpp_mex_config();
if opts.Compile
    fastinfo_cpp_mex_compile();
end

addpath(cfg.OutputDir);
addpath(fullfile(cfg.RepoRoot, 'matlab'));
addpath(cfg.TestsDir);

rng(42, 'twister');
results = run_fastinfo_cpp_mex_smoke_tests();
end
