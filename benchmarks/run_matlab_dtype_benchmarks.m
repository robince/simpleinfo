function results = run_matlab_dtype_benchmarks(varargin)
p = inputParser;
p.addParameter('Compile', false, @(x) islogical(x) && isscalar(x));
p.addParameter('ThreadCounts', [1 2 4], @(x) isnumeric(x) && isvector(x) && ~isempty(x));
p.addParameter('Mode', 'quick', @(x) (ischar(x) || isstring(x)) && ismember(lower(string(x)), ["quick" "full"]));
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

inputs = dtype_inputs(mode);
inputs.config_target_seconds = double(opts.TargetSecondsPerDType);
inputs.config_max_repeats = double(opts.MaxRepeats);
cases = benchmark_cases(inputs, dtypeNames, threadCounts, repeats, opts.Warmup);

results = struct();
results.generated_at = char(datetime('now', TimeZone='local', Format='yyyy-MM-dd''T''HH:mm:ssXXX'));
results.backend = 'mex_cpp';
results.available = true;
results.mode = char(mode);
results.thread_counts = threadCounts;
results.repeats = repeats;
results.warmup = opts.Warmup;
results.target_seconds_per_dtype = opts.TargetSecondsPerDType;
results.max_repeats = opts.MaxRepeats;
results.dtypes = dtypeNames;
results.config = inputs.config;
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

function inputs = dtype_inputs(mode)
cfg = dtype_mode_config(mode);
xb = 16;
yb = 8;
zb = 4;
shared = discrete_vector(cfg.scaling_ntrl, yb, 9001);
z = discrete_vector(cfg.scaling_ntrl, zb, 9002);
y = mod(shared + z + discrete_vector(cfg.scaling_ntrl, 2, 9003), yb);
inputs = struct();
inputs.config = cfg;
inputs.xb = xb;
inputs.yb = yb;
inputs.zb = zb;
inputs.seed = 5489;
inputs.x_slice = mod( ...
    shared + 2 * z + yb * discrete_matrix(cfg.scaling_ntrl, cfg.scaling_nx, 2, 9004) + discrete_matrix(cfg.scaling_ntrl, cfg.scaling_nx, 2, 9005), xb);
inputs.y = y;
inputs.z = z;
inputs.x_matched = mod( ...
    discrete_matrix(cfg.scaling_ntrl, cfg.scaling_nmatched, yb, 9010) + ...
    yb * discrete_matrix(cfg.scaling_ntrl, cfg.scaling_nmatched, 2, 9011) + ...
    discrete_matrix(cfg.scaling_ntrl, cfg.scaling_nmatched, 2, 9012), xb);
inputs.y_matched = mod( ...
    discrete_matrix(cfg.scaling_ntrl, cfg.scaling_nmatched, yb, 9010) + ...
    discrete_matrix(cfg.scaling_ntrl, cfg.scaling_nmatched, 2, 9013), yb);
inputs.nperm = cfg.scaling_nperm;
end

function cfg = dtype_mode_config(mode)
if mode == "quick"
    cfg = struct( ...
        'scaling_ntrl', 1024, ...
        'scaling_nx', 512, ...
        'scaling_nmatched', 512, ...
        'scaling_nperm', 128);
elseif mode == "full"
    cfg = struct( ...
        'scaling_ntrl', 2048, ...
        'scaling_nx', 2048, ...
        'scaling_nmatched', 1024, ...
        'scaling_nperm', 256);
else
    error('run_matlab_dtype_benchmarks:Mode', 'Unsupported mode "%s".', mode);
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

function cases = benchmark_cases(inputs, dtypeNames, threadCounts, repeats, warmup)
maxThreads = max(threadCounts);
fastinfo.calcinfo_slice(int16(randi([0, 1], 32, 16)), 2, int16(randi([0, 1], 32, 1)), 2, 'Threads', maxThreads);
cases = struct('dtype', {}, 'operation', {}, 'kind', {}, 'threads', {}, 'repeats', {}, 'seconds', {}, 'speedup_vs_1', {}, 'max_abs_diff_vs_1', {});
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
        targetSecondsPerMeasurement = inputs.config_target_seconds / measurementsPerDType;
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
                repeatsUsed = calibrated_repeats(@() spec.fn(nThreads), warmup, targetSecondsPerMeasurement, inputs.config_max_repeats);
            else
                repeatsUsed = repeats;
            end
            currentTime = median_runtime(@() spec.fn(nThreads), repeatsUsed, warmup);
            if isempty(baseline)
                baseline = current;
                baselineTime = currentTime;
            end
            cases(end + 1) = struct( ... %#ok<AGROW>
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
