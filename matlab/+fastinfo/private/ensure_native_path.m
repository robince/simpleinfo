function outDir = ensure_native_path()
persistent cachedDir

repoRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
outDir = fullfile(repoRoot, 'matlab', 'cpp_mex', 'bin', version('-release'), mexext);
if exist(outDir, 'dir') && (isempty(cachedDir) || ~strcmp(cachedDir, outDir))
    addpath(outDir);
    cachedDir = outDir;
end
end
