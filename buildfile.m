function plan = buildfile
plan = buildplan(localfunctions);
plan.DefaultTasks = "test";
plan("test").Dependencies = "compile";
plan("package").Dependencies = "compile";
end

function compileTask(~)
add_tooling_path();
fastinfo_cpp_mex_compile();
end

function testTask(~)
add_tooling_path();
fastinfo_cpp_mex_test('Compile', false);
add_validation_path();
run_simpleinfo_validation_tests();
end

function packageTask(~)
add_tooling_path();
fastinfo_cpp_mex_package();
end

function cleanTask(~)
add_tooling_path();
fastinfo_cpp_mex_clean();
end

function add_tooling_path()
root = fileparts(mfilename("fullpath"));
addpath(fullfile(root, "matlab", "cpp_mex", "tooling"));
end

function add_validation_path()
root = fileparts(mfilename("fullpath"));
addpath(fullfile(root, "matlab"));
addpath(fullfile(root, "matlab", "tests"));
end
