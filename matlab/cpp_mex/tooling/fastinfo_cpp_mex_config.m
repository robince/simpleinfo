function cfg = fastinfo_cpp_mex_config()
toolingDir = fileparts(mfilename('fullpath'));
cppMexDir = fileparts(toolingDir);
matlabDir = fileparts(cppMexDir);
repoRoot = fileparts(matlabDir);

release = version('-release');
arch = computer('arch');
matlabRootDir = matlabroot;
matlabBinDir = fullfile(matlabRootDir, 'bin', arch);

cxx = mex.getCompilerConfigurations('C++', 'Selected');
if isempty(cxx)
    error('fastinfo_cpp_mex_config:noCompiler', ...
        'No MATLAB C++ compiler selected; run mex -setup C++.');
end

compilerFamily = i_compiler_family(cxx);
[cxxStdFlag, ompCxxFlag, ompLinkFlags, objExt] = ...
    i_compiler_flags(compilerFamily, matlabBinDir);
ompIncludeDir = i_find_omp_include(matlabRootDir, arch);

compilerTag = regexprep(lower(cxx.Name), '[^a-z0-9]+', '_');
cfg = struct();
cfg.RepoRoot        = repoRoot;
cfg.MatlabDir       = matlabDir;
cfg.CppMexDir       = cppMexDir;
cfg.IncludeDir      = fullfile(cppMexDir, 'include');
cfg.SourceDir       = fullfile(cppMexDir, 'src');
cfg.MexDir          = fullfile(cppMexDir, 'mex');
cfg.TestsDir        = fullfile(cppMexDir, 'tests');
cfg.ToolingDir      = toolingDir;
cfg.OutputDir       = fullfile(cppMexDir, 'bin', release, mexext);
cfg.BuildDir        = fullfile(repoRoot, 'build', 'matlab', 'cpp_mex', release, arch, compilerTag);
cfg.Release         = release;
cfg.Arch            = arch;
cfg.MexExt          = mexext;
cfg.MatlabRoot      = matlabRootDir;
cfg.MatlabBinDir    = matlabBinDir;
cfg.OmpIncludeDir   = ompIncludeDir;
cfg.CompilerName    = cxx.Name;
cfg.CompilerVersion = cxx.Version;
cfg.CompilerFamily  = compilerFamily;
cfg.CxxStdFlag      = cxxStdFlag;
cfg.OmpCxxFlag      = ompCxxFlag;
cfg.OmpLinkFlags    = ompLinkFlags;
cfg.ObjExt          = objExt;
end

function family = i_compiler_family(cxx)
name = lower(cxx.Name);
if ~isempty(regexpi(name, 'microsoft visual c\+\+|msvc', 'once'))
    family = 'msvc';
elseif ~isempty(regexpi(name, 'mingw', 'once'))
    family = 'gcc';
elseif ismac && ~isempty(regexpi(name, 'apple|clang', 'once'))
    family = 'apple_clang';
elseif ~isempty(regexpi(name, 'clang', 'once'))
    family = 'clang';
elseif ~isempty(regexpi(name, 'gnu|g\+\+', 'once'))
    family = 'gcc';
else
    family = 'gcc';
end
end

function [cxxStdFlag, ompCxxFlag, ompLinkFlags, objExt] = i_compiler_flags(family, matlabBinDir)
switch family
    case 'msvc'
        cxxStdFlag   = 'COMPFLAGS=$COMPFLAGS /std:c++17';
        ompCxxFlag   = ' /openmp';
        ompLinkFlags = {};
        objExt       = '.obj';
    case {'gcc', 'clang'}
        if ispc
            rpathFlag = '';
        else
            rpathFlag = [' -Wl,-rpath,' matlabBinDir];
        end
        cxxStdFlag   = 'CXXFLAGS=$CXXFLAGS -std=c++17';
        ompCxxFlag   = ' -fopenmp';
        ompLinkFlags = {['LDFLAGS=$LDFLAGS -fopenmp' rpathFlag]};
        objExt       = '.o';
    otherwise
        cxxStdFlag   = 'CXXFLAGS=$CXXFLAGS -std=c++17';
        ompCxxFlag   = ' -Xpreprocessor -fopenmp';
        ompLinkFlags = { ...
            ['LDFLAGS=$LDFLAGS -Wl,-rpath,' matlabBinDir], ...
            ['LINKLIBS=$LINKLIBS -L' matlabBinDir ' -lomp']};
        objExt       = '.o';
end
end

function ompIncludeDir = i_find_omp_include(matlabRootDir, arch)
ompIncludeDir = '';
if ~ismac
    return
end
candidateDirs = { ...
    fullfile(matlabRootDir, 'toolbox', 'eml', 'externalDependency', 'omp', arch, 'include'), ...
    fullfile(matlabRootDir, 'toolbox', 'coder', 'clang_api', 'llvm-include', arch)};
for i = 1:numel(candidateDirs)
    if exist(fullfile(candidateDirs{i}, 'omp.h'), 'file')
        ompIncludeDir = candidateDirs{i};
        return
    end
end
warning('fastinfo_cpp_mex_config:ompNotFound', ...
    'Could not find MATLAB-shipped omp.h; OpenMP targets may fail to build.');
end
