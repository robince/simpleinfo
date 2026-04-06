function setup_simpleinfo(varargin)
%SETUP_SIMPLEINFO Add simpleinfo MATLAB paths for local use.
%   setup_simpleinfo() adds the public MATLAB API under matlab/ to the path.
%   The fastinfo wrappers add the compiled MEX output directory on demand, so
%   users normally only need this helper.
%
%   setup_simpleinfo(Tests=true) also adds matlab/tests.
%   setup_simpleinfo(Tooling=true) also adds matlab/cpp_mex/tooling.
%   setup_simpleinfo(Save=true) also calls savepath().

p = inputParser;
p.addParameter('Tests', false, @(v) islogical(v) && isscalar(v));
p.addParameter('Tooling', false, @(v) islogical(v) && isscalar(v));
p.addParameter('Save', false, @(v) islogical(v) && isscalar(v));
p.parse(varargin{:});
opts = p.Results;

repoRoot = fileparts(mfilename('fullpath'));
paths = {fullfile(repoRoot, 'matlab')};
if opts.Tests
    paths{end + 1} = fullfile(repoRoot, 'matlab', 'tests'); %#ok<AGROW>
end
if opts.Tooling
    paths{end + 1} = fullfile(repoRoot, 'matlab', 'cpp_mex', 'tooling'); %#ok<AGROW>
end

for i = 1:numel(paths)
    addpath(paths{i});
end

if opts.Save
    savepath();
end
end
