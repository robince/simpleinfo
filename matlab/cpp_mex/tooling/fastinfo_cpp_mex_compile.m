function artifacts = fastinfo_cpp_mex_compile(varargin)
p = inputParser;
p.addParameter('Targets', {}, @(x) iscell(x) || isstring(x) || ischar(x));
p.addParameter('Verbose', false, @(x) islogical(x) && isscalar(x));
p.parse(varargin{:});
opts = p.Results;

cfg = fastinfo_cpp_mex_config();
if ischar(opts.Targets) || isstring(opts.Targets)
    opts.Targets = cellstr(opts.Targets);
end

targets = target_specs(cfg);
if isempty(opts.Targets)
    requested = {targets.name};
else
    requested = cellfun(@char, opts.Targets, 'UniformOutput', false);
end

if ~exist(cfg.BuildDir, 'dir')
    mkdir(cfg.BuildDir);
end
if ~exist(cfg.OutputDir, 'dir')
    mkdir(cfg.OutputDir);
end

artifacts = cell(1, numel(requested));
for i = 1:numel(requested)
    idx = find(strcmp(requested{i}, {targets.name}), 1);
    if isempty(idx)
        error('fastinfo_cpp_mex_compile:unknownTarget', ...
            'Unknown target "%s".', requested{i});
    end
    spec = targets(idx);
    mexArgs = { ...
        '-outdir', cfg.OutputDir, ...
        '-output', spec.name, ...
        '-R2018a', ...
        ['-I' cfg.IncludeDir], ...
        ['-I' cfg.SourceDir]};
    if spec.use_openmp
        if ~isempty(cfg.OmpIncludeDir)
            mexArgs{end + 1} = ['-I' cfg.OmpIncludeDir]; %#ok<AGROW>
        end
        mexArgs{end + 1} = [cfg.CxxStdFlag cfg.OmpCxxFlag]; %#ok<AGROW>
        for k = 1:numel(cfg.OmpLinkFlags)
            mexArgs{end + 1} = cfg.OmpLinkFlags{k}; %#ok<AGROW>
        end
    else
        mexArgs{end + 1} = cfg.CxxStdFlag; %#ok<AGROW>
    end
    if opts.Verbose
        mexArgs{end + 1} = '-v'; %#ok<AGROW>
    end
    mexArgs = [mexArgs, spec.sources]; %#ok<AGROW>
    mex(mexArgs{:});
    artifacts{i} = fullfile(cfg.OutputDir, [spec.name '.' cfg.MexExt]);
end
end

function targets = target_specs(cfg)
targets = struct('name', {}, 'sources', {}, 'use_openmp', {});
targets(end+1) = make_target(cfg, 'fastinfo_calcinfo_cpp', {'fastinfo_calcinfo_cpp.cpp', 'fastinfo_kernels.cpp'}, false);
targets(end+1) = make_target(cfg, 'fastinfo_calcinfomatched_cpp', {'fastinfo_calcinfomatched_cpp.cpp', 'fastinfo_kernels.cpp'}, true);
targets(end+1) = make_target(cfg, 'fastinfo_calccmi_cpp', {'fastinfo_calccmi_cpp.cpp', 'fastinfo_kernels.cpp'}, false);
targets(end+1) = make_target(cfg, 'fastinfo_calccondcmi_cpp', {'fastinfo_calccondcmi_cpp.cpp', 'fastinfo_kernels.cpp'}, false);
targets(end+1) = make_target(cfg, 'fastinfo_calccmi_slice_cpp', {'fastinfo_calccmi_slice_cpp.cpp', 'fastinfo_kernels.cpp'}, true);
targets(end+1) = make_target(cfg, 'fastinfo_calcinfoperm_cpp', {'fastinfo_calcinfoperm_cpp.cpp', 'fastinfo_kernels.cpp'}, true);
targets(end+1) = make_target(cfg, 'fastinfo_calcinfoperm_slice_cpp', {'fastinfo_calcinfoperm_slice_cpp.cpp', 'fastinfo_kernels.cpp'}, true);
targets(end+1) = make_target(cfg, 'fastinfo_calcinfo_slice_cpp', {'fastinfo_calcinfo_slice_cpp.cpp', 'fastinfo_kernels.cpp'}, true);
targets(end+1) = make_target(cfg, 'fastinfo_eqpop_cpp', {'fastinfo_eqpop_cpp.cpp', 'fastinfo_kernels.cpp'}, false);
targets(end+1) = make_target(cfg, 'fastinfo_eqpop_sorted_cpp', {'fastinfo_eqpop_sorted_cpp.cpp', 'fastinfo_kernels.cpp'}, false);
targets(end+1) = make_target(cfg, 'fastinfo_eqpop_slice_cpp', {'fastinfo_eqpop_slice_cpp.cpp', 'fastinfo_kernels.cpp'}, true);
targets(end+1) = make_target(cfg, 'fastinfo_eqpop_sorted_slice_cpp', {'fastinfo_eqpop_sorted_slice_cpp.cpp', 'fastinfo_kernels.cpp'}, true);
end

function target = make_target(cfg, name, fileNames, useOpenMP)
target = struct();
target.name = name;
target.sources = cellfun(@(x) i_resolve_source(cfg, x), fileNames, 'UniformOutput', false);
target.use_openmp = useOpenMP;
end

function pathName = i_resolve_source(cfg, fileName)
if strcmp(fileName, 'fastinfo_kernels.cpp')
    pathName = fullfile(cfg.SourceDir, fileName);
else
    pathName = fullfile(cfg.MexDir, fileName);
end
end
