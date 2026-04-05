function fastinfo_cpp_mex_clean()
cfg = fastinfo_cpp_mex_config();
if exist(cfg.OutputDir, 'dir')
    rmdir(cfg.OutputDir, 's');
end
if exist(cfg.BuildDir, 'dir')
    rmdir(cfg.BuildDir, 's');
end
end
