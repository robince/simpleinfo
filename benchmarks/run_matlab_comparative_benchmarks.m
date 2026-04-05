function results = run_matlab_comparative_benchmarks(varargin)
p = inputParser;
p.addParameter('Compile', false, @(x) islogical(x) && isscalar(x));
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
        fprintf('run_matlab_comparative_benchmarks: using existing MEX runtime in %s\n', runtime.output_dir);
    else
        fprintf('run_matlab_comparative_benchmarks: missing runtime artifacts, will compile.\n');
    end
end
if opts.Compile || ~runtime.available
    fastinfo_cpp_mex_compile();
    runtime = runtime_status();
end
cfgMex = fastinfo_cpp_mex_config();
addpath(cfgMex.OutputDir);

threadCounts = unique(max(1, round(double(opts.ThreadCounts(:)'))));
if isempty(opts.Repeats)
    if mode == "quick"
        repeats = 3;
    else
        repeats = 5;
    end
else
    repeats = double(opts.Repeats);
end

results = struct();
results.generated_at = char(datetime('now', TimeZone='local', Format='yyyy-MM-dd''T''HH:mm:ssXXX'));
results.mode = char(mode);
results.backend = 'mex_cpp';
results.thread_counts = threadCounts;
results.repeats = repeats;
results.warmup = opts.Warmup;
results.config = mode_config(mode);
results.equivalence = run_equivalence(mode);
results.scaling = run_scaling(mode, threadCounts, repeats, opts.Warmup);

outputFile = char(opts.OutputFile);
if isempty(outputFile)
    outputDir = fullfile(repoRoot, 'build', 'benchmarks');
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    stamp = char(datetime('now', Format='yyyyMMdd_HHmmss'));
    outputFile = fullfile(outputDir, sprintf('matlab_fastinfo_comparative_%s.json', stamp));
end
fid = fopen(outputFile, 'w');
if fid < 0
    error('run_matlab_comparative_benchmarks:OutputOpenFailed', 'Unable to open "%s" for writing.', outputFile);
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

function cfg = mode_config(mode)
if mode == "quick"
    cfg = struct( ...
        'equiv_ntrl', 768, ...
        'equiv_nx', 24, ...
        'equiv_nmatched', 16, ...
        'equiv_nperm', 24, ...
        'scaling_ntrl', 2048, ...
        'scaling_nx', 192, ...
        'scaling_nmatched', 128, ...
        'scaling_nperm', 96);
elseif mode == "full"
    cfg = struct( ...
        'equiv_ntrl', 2048, ...
        'equiv_nx', 64, ...
        'equiv_nmatched', 48, ...
        'equiv_nperm', 64, ...
        'scaling_ntrl', 6144, ...
        'scaling_nx', 512, ...
        'scaling_nmatched', 320, ...
        'scaling_nperm', 256);
else
    error('run_matlab_comparative_benchmarks:Mode', 'Unsupported mode "%s".', mode);
end
end

function out = discrete_vector(ntrl, nbins, seed)
out = reshape(mod(hash_grid(ntrl, 1, seed), nbins), ntrl, 1);
out = int16(out);
end

function out = discrete_matrix(ntrl, npage, nbins, seed)
out = int16(mod(hash_grid(ntrl, npage, seed), nbins));
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

function out = permutation_bank(nperm, ntrl, seed)
base = 0:(ntrl - 1);
out = zeros(nperm, ntrl);
for perm = 0:(nperm - 1)
    if ntrl == 1
        a = 1;
    else
        a = mod(2 * mod(seed + 17 * (perm + 1), ntrl) + 1, ntrl);
        if a == 0
            a = 1;
        end
        while gcd(a, ntrl) ~= 1
            a = a + 2;
            if a >= ntrl
                a = mod(a, ntrl);
                if a == 0
                    a = 1;
                end
            end
        end
    end
    b = mod(seed + 97 * (perm + 1) + 13 * perm * perm, max(1, ntrl));
    out(perm + 1, :) = mod(a * base + b, ntrl);
end
end

function inputs = build_equivalence_inputs(mode)
cfg = mode_config(mode);
xb = 16;
yb = 8;
zb = 4;

mi_none_y = discrete_vector(cfg.equiv_ntrl, yb, 2001);
mi_none = struct();
mi_none.x_scalar = discrete_vector(cfg.equiv_ntrl, xb, 1001);
mi_none.y = mi_none_y;
mi_none.x_slice = discrete_matrix(cfg.equiv_ntrl, cfg.equiv_nx, xb, 3001);
mi_none.x_matched = discrete_matrix(cfg.equiv_ntrl, cfg.equiv_nmatched, xb, 4001);
mi_none.y_matched = discrete_matrix(cfg.equiv_ntrl, cfg.equiv_nmatched, yb, 5001);

mi_shared = discrete_vector(cfg.equiv_ntrl, yb, 6001);
mi_effect = struct();
mi_effect.y = mod(mi_shared + mod(discrete_vector(cfg.equiv_ntrl, 2, 6002), 2), yb);
mi_effect.x_scalar = mod(mi_shared + yb * discrete_vector(cfg.equiv_ntrl, 2, 6003) + discrete_vector(cfg.equiv_ntrl, 2, 6004), xb);
mi_effect.x_slice = mod( ...
    mi_shared + ...
    yb * discrete_matrix(cfg.equiv_ntrl, cfg.equiv_nx, 2, 6005) + ...
    discrete_matrix(cfg.equiv_ntrl, cfg.equiv_nx, 2, 6006), xb);
mi_shared_match = discrete_matrix(cfg.equiv_ntrl, cfg.equiv_nmatched, yb, 6101);
mi_effect.x_matched = mod( ...
    mi_shared_match + ...
    yb * discrete_matrix(cfg.equiv_ntrl, cfg.equiv_nmatched, 2, 6102) + ...
    discrete_matrix(cfg.equiv_ntrl, cfg.equiv_nmatched, 2, 6103), xb);
mi_effect.y_matched = mod(mi_shared_match + discrete_matrix(cfg.equiv_ntrl, cfg.equiv_nmatched, 2, 6104), yb);
mi_effect.y = int16(mi_effect.y);
mi_effect.x_scalar = int16(mi_effect.x_scalar);
mi_effect.x_slice = int16(mi_effect.x_slice);
mi_effect.x_matched = int16(mi_effect.x_matched);
mi_effect.y_matched = int16(mi_effect.y_matched);

cmi_none_z = discrete_vector(cfg.equiv_ntrl, zb, 7001);
cmi_none = struct();
cmi_none.x_scalar = int16(mod(3 * cmi_none_z + discrete_vector(cfg.equiv_ntrl, xb, 7002), xb));
cmi_none.y = int16(mod(2 * cmi_none_z + discrete_vector(cfg.equiv_ntrl, yb, 7003), yb));
cmi_none.z = cmi_none_z;
cmi_none.x_slice = int16(mod(3 * cmi_none_z + discrete_matrix(cfg.equiv_ntrl, cfg.equiv_nx, xb, 7004), xb));

cmi_effect_z = discrete_vector(cfg.equiv_ntrl, zb, 8001);
cmi_shared = discrete_vector(cfg.equiv_ntrl, yb, 8002);
cmi_effect = struct();
cmi_effect.x_scalar = int16(mod( ...
    cmi_shared + 3 * cmi_effect_z + yb * discrete_vector(cfg.equiv_ntrl, 2, 8003) + discrete_vector(cfg.equiv_ntrl, 2, 8004), xb));
cmi_effect.y = int16(mod(cmi_shared + 2 * cmi_effect_z + discrete_vector(cfg.equiv_ntrl, 2, 8005), yb));
cmi_effect.z = cmi_effect_z;
cmi_effect.x_slice = int16(mod( ...
    cmi_shared + 3 * cmi_effect_z + yb * discrete_matrix(cfg.equiv_ntrl, cfg.equiv_nx, 2, 8006) + discrete_matrix(cfg.equiv_ntrl, cfg.equiv_nx, 2, 8007), xb));

inputs = struct();
inputs.mi_none = mi_none;
inputs.mi_effect = mi_effect;
inputs.cmi_none = cmi_none;
inputs.cmi_effect = cmi_effect;
end

function inputs = build_scaling_inputs(mode)
cfg = mode_config(mode);
xb = 16;
yb = 8;
zb = 4;
shared = discrete_vector(cfg.scaling_ntrl, yb, 9001);
z = discrete_vector(cfg.scaling_ntrl, zb, 9002);
inputs = struct();
inputs.x_slice = int16(mod( ...
    shared + 2 * z + yb * discrete_matrix(cfg.scaling_ntrl, cfg.scaling_nx, 2, 9004) + discrete_matrix(cfg.scaling_ntrl, cfg.scaling_nx, 2, 9005), xb));
inputs.y = int16(mod(shared + z + discrete_vector(cfg.scaling_ntrl, 2, 9003), yb));
inputs.z = z;
inputs.x_matched = int16(mod( ...
    discrete_matrix(cfg.scaling_ntrl, cfg.scaling_nmatched, yb, 9010) + ...
    yb * discrete_matrix(cfg.scaling_ntrl, cfg.scaling_nmatched, 2, 9011) + ...
    discrete_matrix(cfg.scaling_ntrl, cfg.scaling_nmatched, 2, 9012), xb));
inputs.y_matched = int16(mod( ...
    discrete_matrix(cfg.scaling_ntrl, cfg.scaling_nmatched, yb, 9010) + ...
    discrete_matrix(cfg.scaling_ntrl, cfg.scaling_nmatched, 2, 9013), yb));
inputs.nperm = cfg.scaling_nperm;
end

function payload = serialize_numeric(values)
arr = double(values);
if isscalar(arr)
    payload = struct('shape', [], 'values', arr);
    return
end
if isvector(arr)
    payload = struct('shape', numel(arr), 'values', reshape(arr, 1, []));
    return
end
payload = struct('shape', size(arr), 'values', reshape(arr.', 1, []));
end

function out = reference_calcinfo_slice(X, xb, y, yb)
out = zeros(1, size(X, 2));
for col = 1:size(X, 2)
    out(col) = calcinfo(X(:, col), xb, y, yb, false, 0);
end
end

function diff = max_abs_diff(a, b)
diff = max(abs(double(a(:)) - double(b(:))));
end

function cases = run_equivalence(mode)
inputs = build_equivalence_inputs(mode);
xb = 16;
yb = 8;
zb = 4;

specs = { ...
    {'mi_none', 'calcinfo', @() calcinfo(inputs.mi_none.x_scalar, xb, inputs.mi_none.y, yb, false, 0), @() fastinfo.calcinfo(inputs.mi_none.x_scalar, xb, inputs.mi_none.y, yb)}, ...
    {'mi_effect', 'calcinfo', @() calcinfo(inputs.mi_effect.x_scalar, xb, inputs.mi_effect.y, yb, false, 0), @() fastinfo.calcinfo(inputs.mi_effect.x_scalar, xb, inputs.mi_effect.y, yb)}, ...
    {'mi_none', 'calcinfo_slice', @() reference_calcinfo_slice(inputs.mi_none.x_slice, xb, inputs.mi_none.y, yb), @() fastinfo.calcinfo_slice(inputs.mi_none.x_slice, xb, inputs.mi_none.y, yb)}, ...
    {'mi_effect', 'calcinfo_slice', @() reference_calcinfo_slice(inputs.mi_effect.x_slice, xb, inputs.mi_effect.y, yb), @() fastinfo.calcinfo_slice(inputs.mi_effect.x_slice, xb, inputs.mi_effect.y, yb)}, ...
    {'mi_none', 'calcinfomatched', @() calcinfomatched(inputs.mi_none.x_matched, xb, inputs.mi_none.y_matched, yb, false, 0), @() fastinfo.calcinfomatched(inputs.mi_none.x_matched, xb, inputs.mi_none.y_matched, yb)}, ...
    {'mi_effect', 'calcinfomatched', @() calcinfomatched(inputs.mi_effect.x_matched, xb, inputs.mi_effect.y_matched, yb, false, 0), @() fastinfo.calcinfomatched(inputs.mi_effect.x_matched, xb, inputs.mi_effect.y_matched, yb)}, ...
    {'cmi_none', 'calccmi', @() calccmi(inputs.cmi_none.x_scalar, xb, inputs.cmi_none.y, yb, inputs.cmi_none.z, zb, false, 0), @() fastinfo.calccmi(inputs.cmi_none.x_scalar, xb, inputs.cmi_none.y, yb, inputs.cmi_none.z, zb)}, ...
    {'cmi_effect', 'calccmi', @() calccmi(inputs.cmi_effect.x_scalar, xb, inputs.cmi_effect.y, yb, inputs.cmi_effect.z, zb, false, 0), @() fastinfo.calccmi(inputs.cmi_effect.x_scalar, xb, inputs.cmi_effect.y, yb, inputs.cmi_effect.z, zb)}, ...
    {'cmi_none', 'calccmi_slice', @() calccmi_slice(inputs.cmi_none.x_slice, xb, inputs.cmi_none.y, yb, inputs.cmi_none.z, zb, false, 0), @() fastinfo.calccmi_slice(inputs.cmi_none.x_slice, xb, inputs.cmi_none.y, yb, inputs.cmi_none.z, zb)}, ...
    {'cmi_effect', 'calccmi_slice', @() calccmi_slice(inputs.cmi_effect.x_slice, xb, inputs.cmi_effect.y, yb, inputs.cmi_effect.z, zb, false, 0), @() fastinfo.calccmi_slice(inputs.cmi_effect.x_slice, xb, inputs.cmi_effect.y, yb, inputs.cmi_effect.z, zb)} ...
    };

cases = repmat(struct('scenario', '', 'operation', '', 'naive', struct(), 'fast', struct(), 'max_abs_diff', 0), 1, numel(specs));
for i = 1:numel(specs)
    naive = specs{i}{3}();
    fast = specs{i}{4}();
    cases(i).scenario = specs{i}{1};
    cases(i).operation = specs{i}{2};
    cases(i).naive = serialize_numeric(naive);
    cases(i).fast = serialize_numeric(fast);
    cases(i).max_abs_diff = max_abs_diff(naive, fast);
end
end

function scaling = run_scaling(mode, threadCounts, repeats, warmup)
inputs = build_scaling_inputs(mode);
xb = 16;
yb = 8;
zb = 4;
seed = 5489;
caseSpecs = { ...
    {'calcinfo_slice', @(nThreads) fastinfo.calcinfo_slice(inputs.x_slice, xb, inputs.y, yb, 'Threads', nThreads)}, ...
    {'calcinfomatched', @(nThreads) fastinfo.calcinfomatched(inputs.x_matched, xb, inputs.y_matched, yb, 'Threads', nThreads)}, ...
    {'calccmi_slice', @(nThreads) fastinfo.calccmi_slice(inputs.x_slice, xb, inputs.y, yb, inputs.z, zb, 'Threads', nThreads)}, ...
    {'calcinfoperm_slice', @(nThreads) fastinfo.calcinfoperm_slice(inputs.x_slice, xb, inputs.y, yb, inputs.nperm, 'Threads', nThreads, 'Seed', seed)} ...
    };

entries = struct('operation', {}, 'threads', {}, 'seconds', {}, 'speedup_vs_1', {}, 'max_abs_diff_vs_1', {});
for i = 1:numel(caseSpecs)
    baseline = [];
    baselineTime = [];
    for j = 1:numel(threadCounts)
        nThreads = threadCounts(j);
        current = caseSpecs{i}{2}(nThreads);
        currentTime = median_runtime(@() caseSpecs{i}{2}(nThreads), repeats, warmup);
        if isempty(baseline)
            baseline = current;
            baselineTime = currentTime;
        end
        entries(end + 1) = struct( ...
            'operation', caseSpecs{i}{1}, ...
            'threads', nThreads, ...
            'seconds', currentTime, ...
            'speedup_vs_1', baselineTime / currentTime, ...
            'max_abs_diff_vs_1', max_abs_diff(current, baseline)); %#ok<AGROW>
    end
end
scaling = struct('available', true, 'backend', 'mex_cpp', 'cases', entries);
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
