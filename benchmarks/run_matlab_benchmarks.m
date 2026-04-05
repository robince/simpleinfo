function results = run_matlab_benchmarks(varargin)
p = inputParser;
p.addParameter('Compile', false, @(x) islogical(x) && isscalar(x));
p.addParameter('LegacyRepo', '/Users/robince/Dropbox (Personal)/Mac/Documents/glacode/info', @(x) ischar(x) || isstring(x));
p.addParameter('ThreadCounts', [1 2 4], @(x) isnumeric(x) && isvector(x) && ~isempty(x));
p.addParameter('Mode', 'quick', @(x) (ischar(x) || isstring(x)) && ismember(lower(string(x)), ["quick" "full"]));
p.addParameter('Repeats', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x >= 1));
p.addParameter('Warmup', 1, @(x) isnumeric(x) && isscalar(x) && x >= 0);
p.addParameter('Verbose', true, @(x) islogical(x) && isscalar(x));
p.addParameter('OutputFile', '', @(x) ischar(x) || isstring(x));
p.parse(varargin{:});
opts = p.Results;
mode = lower(string(opts.Mode));

repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repoRoot, 'matlab'));
addpath(fullfile(repoRoot, 'matlab', 'cpp_mex', 'tooling'));

runtime = runtime_status();
if opts.Verbose
    if runtime.available
        fprintf('run_matlab_benchmarks: using existing MEX runtime in %s\n', runtime.output_dir);
    else
        fprintf('run_matlab_benchmarks: missing runtime artifacts, will compile.\n');
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

legacyRepo = char(opts.LegacyRepo);
legacyAvailable = exist(legacyRepo, 'dir') == 7;
if legacyAvailable
    addpath(legacyRepo);
end

threadCounts = unique(max(1, round(double(opts.ThreadCounts(:)'))));
rng(42, 'twister');

if isempty(opts.Repeats)
    if mode == "quick"
        repeats = 3;
    else
        repeats = 7;
    end
else
    repeats = double(opts.Repeats);
end

cases = benchmark_cases(threadCounts, mode);
if isempty(cases)
    caseResults = struct([]);
else
    caseResults(1) = run_case(cases(1), threadCounts, legacyAvailable, repeats, opts.Warmup); %#ok<AGROW>
    for i = 2:numel(cases)
        caseResults(i) = run_case(cases(i), threadCounts, legacyAvailable, repeats, opts.Warmup); %#ok<AGROW>
    end
end

results = struct();
results.generated_at = char(datetime('now', TimeZone='local', Format='yyyy-MM-dd''T''HH:mm:ssXXX'));
results.matlab_release = cfg.Release;
results.arch = cfg.Arch;
results.mexext = cfg.MexExt;
results.compiler = cfg.CompilerName;
results.legacy_repo = legacyRepo;
results.legacy_available = legacyAvailable;
results.thread_counts = threadCounts;
results.mode = char(mode);
results.repeats = repeats;
results.warmup = opts.Warmup;
results.cases = caseResults;

outputFile = char(opts.OutputFile);
if isempty(outputFile)
    outputDir = fullfile(repoRoot, 'build', 'benchmarks');
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    stamp = char(datetime('now', Format='yyyyMMdd_HHmmss'));
    outputFile = fullfile(outputDir, sprintf('fastinfo_benchmarks_%s.json', stamp));
end

fid = fopen(outputFile, 'w');
if fid < 0
    error('run_matlab_benchmarks:OutputOpenFailed', ...
        'Unable to open "%s" for writing.', outputFile);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(results, PrettyPrint=true));
fprintf('Wrote benchmark results to %s\n', outputFile);
end

function runtime = runtime_status()
cfg = fastinfo_cpp_mex_config();
required = { ...
    'fastinfo_calcinfo_cpp', ...
    'fastinfo_calcinfomatched_cpp', ...
    'fastinfo_calccmi_cpp', ...
    'fastinfo_calccondcmi_cpp', ...
    'fastinfo_calccmi_slice_cpp', ...
    'fastinfo_calcinfoperm_cpp', ...
    'fastinfo_calcinfoperm_slice_cpp', ...
    'fastinfo_calcinfo_slice_cpp', ...
    'fastinfo_eqpop_cpp', ...
    'fastinfo_eqpop_sorted_cpp', ...
    'fastinfo_eqpop_slice_cpp', ...
    'fastinfo_eqpop_sorted_slice_cpp'};
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

function cases = benchmark_cases(threadCounts, mode)
maxThreads = max(threadCounts);

cases = struct('name', {}, 'kind', {}, 'fast_fn', {}, 'reference_fn', {}, 'reference_supported', {}, 'legacy_fn', {}, 'legacy_supported', {});

if mode == "quick"
    Ntrl = 1500;
    Nx = 192;
    Xm = 16;
    Ym = 8;
    permNtrl = 1500;
    permCount = 96;
    eqpopN = 40000;
else
    Ntrl = 4000;
    Nx = 512;
    Xm = 16;
    Ym = 8;
    permNtrl = 4000;
    permCount = 256;
    eqpopN = 200000;
end

X = int16(randi([0, Xm - 1], Ntrl, Nx));
Y = int16(randi([0, Ym - 1], Ntrl, 1));
cases(end + 1) = struct( ...
    'name', 'calcinfo_slice_large', ...
    'kind', 'threaded', ...
    'fast_fn', @(nThreads) fastinfo.calcinfo_slice(X, Xm, Y, Ym, 'Threads', nThreads), ...
    'reference_fn', @() i_calcinfo_slice_reference(X, Xm, Y, Ym), ...
    'reference_supported', @() true, ...
    'legacy_fn', @(nThreads) info.calc_info_slice_omp_integer_c_int16_t(X, Xm, Y, Ym, Ntrl, nThreads), ...
    'legacy_supported', @() exist('info.calc_info_slice_omp_integer_c_int16_t', 'file') == 3);

Ymatched = int16(randi([0, Ym - 1], Ntrl, Nx));
cases(end + 1) = struct( ...
    'name', 'calcinfomatched_large', ...
    'kind', 'threaded', ...
    'fast_fn', @(nThreads) fastinfo.calcinfomatched(X, Xm, Ymatched, Ym, 'Threads', nThreads), ...
    'reference_fn', @() calcinfomatched(X, Xm, Ymatched, Ym, false, 0), ...
    'reference_supported', @() true, ...
    'legacy_fn', @(nThreads) info.calc_info_omp_integer_c_int16_t(X, Xm, Ymatched, Ym, Ntrl, nThreads), ...
    'legacy_supported', @() exist('info.calc_info_omp_integer_c_int16_t', 'file') == 3);

Ntrl = 512;
Xm = 8;
Ym = 8;
x = int16(randi([0, Xm - 1], Ntrl, 1));
y = int16(randi([0, Ym - 1], Ntrl, 1));
cases(end + 1) = struct( ...
    'name', 'calcinfo_scalar', ...
    'kind', 'scalar', ...
    'fast_fn', @(~) fastinfo.calcinfo(x, Xm, y, Ym), ...
    'reference_fn', @() calcinfo(x, Xm, y, Ym, false, 0), ...
    'reference_supported', @() true, ...
    'legacy_fn', @(~) info.calc_info_integer_c_int16_t(x, Xm, y, Ym, Ntrl), ...
    'legacy_supported', @() exist('info.calc_info_integer_c_int16_t', 'file') == 3);

z = int16(randi([0, 3], Ntrl, 1));
Xcmi = int16(randi([0, Xm - 1], Ntrl, 96));
cases(end + 1) = struct( ...
    'name', 'calccmi_slice', ...
    'kind', 'threaded', ...
    'fast_fn', @(nThreads) fastinfo.calccmi_slice(Xcmi, Xm, y, Ym, z, 4, 'Threads', nThreads), ...
    'reference_fn', @() calccmi_slice(Xcmi, Xm, y, Ym, z, 4, false, 0), ...
    'reference_supported', @() true, ...
    'legacy_fn', @(~) [], ...
    'legacy_supported', @() false);

Ntrl = permNtrl;
Xm = 16;
Ym = 8;
x = int16(randi([0, Xm - 1], Ntrl, 1));
y = int16(randi([0, Ym - 1], Ntrl, 1));
nPerm = permCount;
cases(end + 1) = struct( ...
    'name', 'calcinfoperm', ...
    'kind', 'threaded', ...
    'fast_fn', @(nThreads) fastinfo.calcinfoperm(x, Xm, y, Ym, nPerm, 'Threads', nThreads), ...
    'reference_fn', @() calcinfoperm(x, Xm, y, Ym, nPerm, false, 0), ...
    'reference_supported', @() true, ...
    'legacy_fn', @(nThreads) info.calc_info_bootstrap_omp_integer_c_int16_t(x, Xm, y, Ym, Ntrl, nPerm, nThreads), ...
    'legacy_supported', @() exist('info.calc_info_bootstrap_omp_integer_c_int16_t', 'file') == 3);

Xperm = int16(randi([0, Xm - 1], Ntrl, 48));
cases(end + 1) = struct( ...
    'name', 'calcinfoperm_slice', ...
    'kind', 'threaded', ...
    'fast_fn', @(nThreads) fastinfo.calcinfoperm_slice(Xperm, Xm, y, Ym, nPerm, 'Threads', nThreads), ...
    'reference_fn', @() calcinfoperm_slice(Xperm, Xm, y, Ym, nPerm, false, 0), ...
    'reference_supported', @() true, ...
    'legacy_fn', @(~) [], ...
    'legacy_supported', @() false);

N = eqpopN;
nb = 8;
x = sort(randn(N, 1));
cases(end + 1) = struct( ...
    'name', 'eqpop_sorted', ...
    'kind', 'scalar', ...
    'fast_fn', @(~) fastinfo.eqpop_sorted(x, nb), ...
    'reference_fn', @() i_eqpop_sorted_reference_benchmark(x, nb), ...
    'reference_supported', @() true, ...
    'legacy_fn', @(~) bin.eqpop_sorted(x, nb), ...
    'legacy_supported', @() exist('bin.eqpop_sorted', 'file') == 3);

Xeq = randn(min(eqpopN, 40000), 32);
Xeqs = sort(Xeq, 1);
cases(end + 1) = struct( ...
    'name', 'eqpop_slice', ...
    'kind', 'threaded', ...
    'fast_fn', @(nThreads) fastinfo.eqpop_slice(Xeq, nb, 'Threads', nThreads), ...
    'reference_fn', @() i_eqpop_slice_reference(Xeq, nb), ...
    'reference_supported', @() true, ...
    'legacy_fn', @(~) [], ...
    'legacy_supported', @() false);
cases(end + 1) = struct( ...
    'name', 'eqpop_sorted_slice', ...
    'kind', 'threaded', ...
    'fast_fn', @(nThreads) fastinfo.eqpop_sorted_slice(Xeqs, nb, 'Threads', nThreads), ...
    'reference_fn', @() i_eqpop_sorted_slice_reference(Xeqs, nb), ...
    'reference_supported', @() true, ...
    'legacy_fn', @(~) [], ...
    'legacy_supported', @() false);

% Warm thread pool once so timeit does not include first-use thread startup.
fastinfo.calcinfo_slice(int16(randi([0, 1], 32, 16)), 2, int16(randi([0, 1], 32, 1)), 2, 'Threads', maxThreads);
end

function result = run_case(spec, threadCounts, legacyAvailable, repeats, warmup)
result = struct();
result.name = spec.name;
result.kind = spec.kind;
result.fast = measure_impl(spec.fast_fn, threadCounts, strcmp(spec.kind, 'threaded'), repeats, warmup);
result.fast.scaling = compute_scaling(result.fast.timings);

if spec.reference_supported()
    result.reference = measure_impl(@(~) spec.reference_fn(), 1, false, repeats, warmup);
    result.reference.scaling = compute_scaling(result.reference.timings);
else
    result.reference = struct('available', false, 'thread_counts', [], 'timings', [], 'scaling', []);
end

if legacyAvailable && spec.legacy_supported()
    result.legacy = measure_impl(spec.legacy_fn, threadCounts, strcmp(spec.kind, 'threaded'), repeats, warmup);
    result.legacy.scaling = compute_scaling(result.legacy.timings);
else
    result.legacy = struct('available', false, 'thread_counts', [], 'timings', [], 'scaling', []);
end
end

function out = measure_impl(fn, threadCounts, threaded, repeats, warmup)
if threaded
    useThreads = threadCounts;
else
    useThreads = threadCounts(1);
end

timings = zeros(size(useThreads));
for i = 1:numel(useThreads)
    for j = 1:warmup
        fn(useThreads(i));
    end
    timings(i) = fixed_repeat_time(@() fn(useThreads(i)), repeats);
end
out = struct('available', true, 'thread_counts', useThreads, 'timings', timings);
end

function t = fixed_repeat_time(fn, repeats)
elapsed = zeros(1, repeats);
for i = 1:repeats
    tic;
    fn();
    elapsed(i) = toc;
end
t = median(elapsed);
end

function scaling = compute_scaling(timings)
if isempty(timings)
    scaling = [];
    return
end
scaling = timings(1) ./ timings;
end

function out = i_calcinfo_slice_reference(X, Xm, Y, Ym)
out = zeros(size(X, 2), 1);
for col = 1:size(X, 2)
    out(col) = calcinfo(X(:, col), Xm, Y, Ym, false, 0);
end
end

function out = i_eqpop_sorted_reference_benchmark(Xs, nb)
out = i_eqpop_sorted_reference(Xs, nb);
end

function out = i_eqpop_slice_reference(X, nb)
out = nan(size(X));
for col = 1:size(X, 2)
    try
        out(:, col) = double(i_eqpop_reference(X(:, col), nb));
    catch
        out(:, col) = nan(size(X, 1), 1);
    end
end
end

function out = i_eqpop_sorted_slice_reference(Xs, nb)
out = nan(size(Xs));
for col = 1:size(Xs, 2)
    try
        out(:, col) = double(i_eqpop_sorted_reference(Xs(:, col), nb));
    catch
        out(:, col) = nan(size(Xs, 1), 1);
    end
end
end

function out = i_eqpop_reference(x, nb)
[xSorted, order] = sort(x(:), 'ascend');
xSortedBin = i_eqpop_sorted_reference(xSorted, nb);
out = zeros(size(xSortedBin));
out(order) = xSortedBin;
out = reshape(out, size(x));
end

function xbin = i_eqpop_sorted_reference(xSorted, nb)
xSorted = xSorted(:);
n = numel(xSorted);
if n < nb
    error('benchmark:eqpop', 'nb cannot exceed the number of samples.');
end
if any(~isfinite(xSorted))
    error('benchmark:eqpop', 'Input must contain only finite values.');
end
if any(diff(xSorted) < 0)
    error('benchmark:eqpop', 'Input must be sorted.');
end

groupStarts = [1; find(diff(xSorted) ~= 0) + 1; n + 1];
nGroups = numel(groupStarts) - 1;
if nGroups < nb
    error('benchmark:eqpop', 'Cannot form the requested number of equal-population bins without splitting tied values.');
end

ideal = n / nb;
dp = inf(nb + 1, nGroups + 1);
parent = zeros(nb + 1, nGroups + 1);
dp(1, 1) = 0;
for b = 1:nb
    minUsed = b;
    maxUsed = nGroups - (nb - b);
    for g = minUsed:maxUsed
        for prev = (b - 1):(g - 1)
            prefix = dp(b, prev + 1);
            if ~isfinite(prefix)
                continue
            end
            count = groupStarts(g + 1) - groupStarts(prev + 1);
            deviation = count - ideal;
            cost = prefix + deviation.^2;
            if cost < dp(b + 1, g + 1)
                dp(b + 1, g + 1) = cost;
                parent(b + 1, g + 1) = prev;
            end
        end
    end
end

cuts = zeros(nb + 1, 1);
cuts(end) = nGroups;
g = nGroups;
for b = nb:-1:1
    prev = parent(b + 1, g + 1);
    if prev < 0
        error('benchmark:eqpop', 'Failed to reconstruct equal-population partition.');
    end
    cuts(b) = prev;
    g = prev;
end

xbin = zeros(n, 1, 'int32');
for b = 1:nb
    startIdx = groupStarts(cuts(b) + 1);
    stopIdx = groupStarts(cuts(b + 1) + 1) - 1;
    xbin(startIdx:stopIdx) = int32(b - 1);
end
end
