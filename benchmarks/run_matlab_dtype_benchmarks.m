function results = run_matlab_dtype_benchmarks(varargin)
p = inputParser;
p.addParameter('Compile', false, @(x) islogical(x) && isscalar(x));
p.addParameter('ThreadCounts', [1 2 4], @(x) isnumeric(x) && isvector(x) && ~isempty(x));
p.addParameter('Mode', 'quick', @(x) (ischar(x) || isstring(x)) && ismember(lower(string(x)), ["quick" "full"]));
p.addParameter('Regimes', {}, @(x) iscellstr(x) || isstring(x));
p.addParameter('Repeats', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x >= 1));
p.addParameter('Warmup', 1, @(x) isnumeric(x) && isscalar(x) && x >= 0);
p.addParameter('TargetSecondsPerDType', 10, @(x) isnumeric(x) && isscalar(x) && x > 0);
p.addParameter('MaxRepeats', 25, @(x) isnumeric(x) && isscalar(x) && x >= 1);
p.addParameter('DTypes', {'int16', 'int32', 'int64'}, @(x) iscellstr(x) || isstring(x));
p.addParameter('Verbose', true, @(x) islogical(x) && isscalar(x));
p.addParameter('OutputFile', '', @(x) ischar(x) || isstring(x));
p.parse(varargin{:});
opts = p.Results;
mode = lower(string(opts.Mode));
regimeNames = resolve_regimes(mode, opts.Regimes);

repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repoRoot, 'matlab'));
addpath(fullfile(repoRoot, 'matlab', 'cpp_mex', 'tooling'));
addpath(fullfile(repoRoot, 'benchmarks'));

runtime = runtime_status();
if opts.Verbose
    if runtime.available
        fprintf('run_matlab_dtype_benchmarks: using existing MEX runtime in %s\n', runtime.output_dir);
    else
        fprintf('run_matlab_dtype_benchmarks: missing runtime artifacts, will compile.\n');
        for i = 1:numel(runtime.missing)
            fprintf('  missing: %s\n', runtime.missing{i});
        end
    end
end
if opts.Compile || ~runtime.available
    fastinfo_cpp_mex_compile();
    runtime = runtime_status();
end
cfg = fastinfo_cpp_mex_config();
addpath(cfg.OutputDir);

threadCounts = unique(max(1, round(double(opts.ThreadCounts(:)'))));
dtypeNames = cellstr(string(opts.DTypes(:)'));
if isempty(opts.Repeats)
    repeats = [];
else
    repeats = double(opts.Repeats);
end

[cases, configs] = benchmark_cases(regimeNames, dtypeNames, threadCounts, repeats, opts.Warmup, double(opts.TargetSecondsPerDType), double(opts.MaxRepeats));

results = struct();
results.generated_at = char(datetime('now', TimeZone='local', Format='yyyy-MM-dd''T''HH:mm:ssXXX'));
results.backend = 'mex_cpp';
results.available = true;
results.mode = char(mode);
results.regimes = cellstr(regimeNames);
results.thread_counts = threadCounts;
results.repeats = repeats;
results.warmup = opts.Warmup;
results.target_seconds_per_dtype = opts.TargetSecondsPerDType;
results.max_repeats = opts.MaxRepeats;
results.dtypes = dtypeNames;
results.configs = configs;
results.cases = cases;

outputFile = char(opts.OutputFile);
if isempty(outputFile)
    outputDir = fullfile(repoRoot, 'build', 'benchmarks');
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    stamp = char(datetime('now', Format='yyyyMMdd_HHmmss'));
    outputFile = fullfile(outputDir, sprintf('matlab_fastinfo_dtypes_%s.json', stamp));
end

fid = fopen(outputFile, 'w');
if fid < 0
    error('run_matlab_dtype_benchmarks:OutputOpenFailed', 'Unable to open "%s" for writing.', outputFile);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(results, PrettyPrint=true));
fprintf('Wrote dtype benchmark results to %s\n', outputFile);
end

function runtime = runtime_status()
cfg = fastinfo_cpp_mex_config();
required = { ...
    'fastinfo_calcinfo_cpp', ...
    'fastinfo_calcinfomatched_cpp', ...
    'fastinfo_calccmi_cpp', ...
    'fastinfo_calcinfo_slice_cpp', ...
    'fastinfo_calccmi_slice_cpp', ...
    'fastinfo_calcinfoperm_cpp', ...
    'fastinfo_calcinfoperm_slice_cpp'};
runtime = struct();
runtime.output_dir = cfg.OutputDir;
runtime.available = exist(cfg.OutputDir, 'dir') == 7;
runtime.missing = {};
if ~runtime.available
    runtime.missing = required;
    return
end
for i = 1:numel(required)
    if exist(fullfile(cfg.OutputDir, [required{i} '.' cfg.MexExt]), 'file') ~= 2
        runtime.available = false;
        runtime.missing{end + 1} = required{i}; %#ok<AGROW>
    end
end
end

function regimeNames = resolve_regimes(mode, regimes)
if ~isempty(regimes)
    regimeNames = lower(string(regimes(:)'));
    return
end
if mode == "quick"
    regimeNames = "medium";
elseif mode == "full"
    regimeNames = "large";
else
    error('run_matlab_dtype_benchmarks:Mode', 'Unsupported mode "%s".', mode);
end
end

function inputs = dtype_inputs(regimeName)
cfg = dtype_regime_config(regimeName);
xb = 16;
yb = 8;
zb = 4;
shared = discrete_vector(cfg.ntrl, yb, 9001);
z = discrete_vector(cfg.ntrl, zb, 9002);
y = mod(shared + z + discrete_vector(cfg.ntrl, 2, 9003), yb);
inputs = struct();
inputs.config = cfg;
inputs.xb = xb;
inputs.yb = yb;
inputs.zb = zb;
inputs.seed = 5489;
inputs.x_slice = mod( ...
    shared + 2 * z + yb * discrete_matrix(cfg.ntrl, cfg.nx, 2, 9004) + discrete_matrix(cfg.ntrl, cfg.nx, 2, 9005), xb);
inputs.y = y;
inputs.z = z;
inputs.x_matched = mod( ...
    discrete_matrix(cfg.ntrl, cfg.nmatched, yb, 9010) + ...
    yb * discrete_matrix(cfg.ntrl, cfg.nmatched, 2, 9011) + ...
    discrete_matrix(cfg.ntrl, cfg.nmatched, 2, 9012), xb);
inputs.y_matched = mod( ...
    discrete_matrix(cfg.ntrl, cfg.nmatched, yb, 9010) + ...
    discrete_matrix(cfg.ntrl, cfg.nmatched, 2, 9013), yb);
inputs.nperm = cfg.nperm;
end

function cfg = dtype_regime_config(regimeName)
regimeName = lower(string(regimeName));
if regimeName == "small"
    cfg = struct( ...
        'ntrl', 256, ...
        'nx', 128, ...
        'nmatched', 128, ...
        'nperm', 64);
elseif regimeName == "medium"
    cfg = struct( ...
        'ntrl', 1024, ...
        'nx', 512, ...
        'nmatched', 512, ...
        'nperm', 128);
elseif regimeName == "large"
    cfg = struct( ...
        'ntrl', 4096, ...
        'nx', 2048, ...
        'nmatched', 1024, ...
        'nperm', 256);
else
    error('run_matlab_dtype_benchmarks:Regime', 'Unsupported regime "%s".', regimeName);
end
end

function out = discrete_vector(ntrl, nbins, seed)
out = reshape(mod(hash_grid(ntrl, 1, seed), nbins), ntrl, 1);
end

function out = discrete_matrix(ntrl, npage, nbins, seed)
out = mod(hash_grid(ntrl, npage, seed), nbins);
end

function out = hash_grid(ntrl, npage, seed)
prime = 2147483647;
t = (1:ntrl)';
p = 1:npage;
out = mod( ...
    double(seed) + ...
    104729 * t + ...
    17077 * p + ...
    433 * (t .* p) + ...
    811 * (t .^ 2) + ...
    233 * (p .^ 2), ...
    prime);
end

function out = cast_discrete(values, dtypeName)
out = cast(values, dtypeName);
end

function [cases, configs] = benchmark_cases(regimeNames, dtypeNames, threadCounts, repeats, warmup, targetSecondsPerDType, maxRepeats)
maxThreads = max(threadCounts);
fastinfo.calcinfo_slice(int16(randi([0, 1], 32, 16)), 2, int16(randi([0, 1], 32, 1)), 2, 'Threads', maxThreads);
cases = struct('regime', {}, 'dtype', {}, 'operation', {}, 'kind', {}, 'threads', {}, 'repeats', {}, 'seconds', {}, 'speedup_vs_1', {}, 'max_abs_diff_vs_1', {});
configs = struct('regime', {}, 'ntrl', {}, 'nx', {}, 'nmatched', {}, 'nperm', {});
for r = 1:numel(regimeNames)
    regimeName = char(regimeNames(r));
    inputs = dtype_inputs(regimeName);
    configs(end + 1) = struct( ... %#ok<AGROW>
        'regime', regimeName, ...
        'ntrl', inputs.config.ntrl, ...
        'nx', inputs.config.nx, ...
        'nmatched', inputs.config.nmatched, ...
        'nperm', inputs.config.nperm);
    for d = 1:numel(dtypeNames)
        dtypeName = dtypeNames{d};
        xScalar = cast_discrete(inputs.x_slice(:, 1), dtypeName);
        y = cast_discrete(inputs.y, dtypeName);
        z = cast_discrete(inputs.z, dtypeName);
        xSlice = cast_discrete(inputs.x_slice, dtypeName);
        xMatched = cast_discrete(inputs.x_matched, dtypeName);
        yMatched = cast_discrete(inputs.y_matched, dtypeName);
        specs = { ...
            struct('operation', 'calcinfo', 'kind', 'scalar', 'fn', @(~) fastinfo.calcinfo(xScalar, inputs.xb, y, inputs.yb)), ...
            struct('operation', 'calcinfo_slice', 'kind', 'threaded', 'fn', @(nThreads) fastinfo.calcinfo_slice(xSlice, inputs.xb, y, inputs.yb, 'Threads', nThreads)), ...
            struct('operation', 'calcinfomatched', 'kind', 'threaded', 'fn', @(nThreads) fastinfo.calcinfomatched(xMatched, inputs.xb, yMatched, inputs.yb, 'Threads', nThreads)), ...
            struct('operation', 'calccmi', 'kind', 'scalar', 'fn', @(~) fastinfo.calccmi(xScalar, inputs.xb, y, inputs.yb, z, inputs.zb)), ...
            struct('operation', 'calccmi_slice', 'kind', 'threaded', 'fn', @(nThreads) fastinfo.calccmi_slice(xSlice, inputs.xb, y, inputs.yb, z, inputs.zb, 'Threads', nThreads)), ...
            struct('operation', 'calcinfoperm', 'kind', 'threaded', 'fn', @(nThreads) fastinfo.calcinfoperm(xScalar, inputs.xb, y, inputs.yb, inputs.nperm, 'Threads', nThreads, 'Seed', inputs.seed)), ...
            struct('operation', 'calcinfoperm_slice', 'kind', 'threaded', 'fn', @(nThreads) fastinfo.calcinfoperm_slice(xSlice, inputs.xb, y, inputs.yb, inputs.nperm, 'Threads', nThreads, 'Seed', inputs.seed)) ...
            };
        for s = 1:numel(specs)
            spec = specs{s};
            measurementsPerDType = numel(threadCounts) * 5 + 2;
            targetSecondsPerMeasurement = targetSecondsPerDType / measurementsPerDType;
            if strcmp(spec.kind, 'threaded')
                useThreads = threadCounts;
            else
                useThreads = threadCounts(1);
            end
            baseline = [];
            baselineTime = [];
            for t = 1:numel(useThreads)
                nThreads = useThreads(t);
                current = spec.fn(nThreads);
                if isempty(repeats)
                    repeatsUsed = calibrated_repeats(@() spec.fn(nThreads), warmup, targetSecondsPerMeasurement, maxRepeats);
                else
                    repeatsUsed = repeats;
                end
                currentTime = median_runtime(@() spec.fn(nThreads), repeatsUsed, warmup);
                if isempty(baseline)
                    baseline = current;
                    baselineTime = currentTime;
                end
                cases(end + 1) = struct( ... %#ok<AGROW>
                    'regime', regimeName, ...
                    'dtype', dtypeName, ...
                    'operation', spec.operation, ...
                    'kind', spec.kind, ...
                    'threads', nThreads, ...
                    'repeats', repeatsUsed, ...
                    'seconds', currentTime, ...
                    'speedup_vs_1', baselineTime / currentTime, ...
                    'max_abs_diff_vs_1', max_abs_diff(current, baseline));
            end
        end
    end
end
end

function repeats = calibrated_repeats(fn, warmup, targetSeconds, maxRepeats)
sample = median_runtime(fn, 1, warmup);
if sample <= 0
    repeats = maxRepeats;
    return
end
repeats = max(1, min(maxRepeats, ceil(targetSeconds / sample)));
end

function elapsed = median_runtime(fn, repeats, warmup)
for i = 1:warmup
    fn();
end
timings = zeros(repeats, 1);
for i = 1:repeats
    tic;
    fn();
    timings(i) = toc;
end
elapsed = median(timings);
end

function diff = max_abs_diff(a, b)
diff = max(abs(double(a(:)) - double(b(:))));
end
