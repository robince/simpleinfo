function results = run_simpleinfo_validation_tests()
results = struct('passed', 0, 'failed', 0, 'details', {{}});
tests = { ...
    @test_numdec2base_exact_power, ...
    @test_numdec2base_width_too_small, ...
    @test_rebin_exact_small_case, ...
    @test_fastinfo_calcinfo_matches_reference, ...
    @test_fastinfo_calcinfomatched_matches_reference, ...
    @test_fastinfo_calccmi_matches_reference, ...
    @test_fastinfo_calccondcmi_matches_reference, ...
    @test_fastinfo_calcinfo_slice_matches_reference, ...
    @test_fastinfo_calccmi_slice_matches_reference, ...
    @test_fastinfo_calcpairwiseinfo_matches_reference};

for i = 1:numel(tests)
    name = func2str(tests{i});
    try
        feval(tests{i});
        results.passed = results.passed + 1;
        results.details{end + 1} = sprintf('PASS %s', name); %#ok<AGROW>
    catch ME
        results.failed = results.failed + 1;
        results.details{end + 1} = sprintf('FAIL %s: %s', name, ME.message); %#ok<AGROW>
    end
end

for i = 1:numel(results.details)
    disp(results.details{i});
end
fprintf('simpleinfo validation summary: %d passed, %d failed\n', results.passed, results.failed);
if results.failed > 0
    error('simpleinfo_validation_tests:failuresDetected', 'simpleinfo validation tests failed.');
end
end

function test_numdec2base_exact_power()
digits = numdec2base([8 9], 2);
expected = [1 1; 0 0; 0 0; 0 1];
assert(isequal(digits, expected));
end

function test_numdec2base_width_too_small()
assert_error_contains(@() numdec2base(8, 2, 3), 'too small');
end

function test_rebin_exact_small_case()
x = [0 1 2 2 3 3];
rebinned = rebin(x, 2);
assert(isequal(size(rebinned), size(x)));
assert(all(rebinned(:) >= 0));
assert(all(rebinned(:) <= 1));
end

function test_fastinfo_calcinfo_matches_reference()
x = int16([0 0 1 1]');
y = int16([0 0 1 1]');
actual = fastinfo.calcinfo(x, 2, y, 2);
expected = calcinfo(x, 2, y, 2, false, 0);
assert(abs(actual - expected) < 1e-12);
end

function test_fastinfo_calcinfomatched_matches_reference()
x = int16([0 0 1; 0 1 1; 1 0 0; 1 1 0]);
y = int16([0 1 1; 0 0 1; 1 1 0; 1 0 0]);
actual = fastinfo.calcinfomatched(x, 2, y, 2);
expected = calcinfomatched(x, 2, y, 2, false, 0);
assert(max(abs(actual(:) - expected(:))) < 1e-12);
end

function test_fastinfo_calccmi_matches_reference()
x = int16([0 0 1 1 0 0 1 1]');
y = int16([0 0 1 1 0 1 0 1]');
z = int16([0 0 0 0 1 1 1 1]');
actual = fastinfo.calccmi(x, 2, y, 2, z, 2);
expected = calccmi(x, 2, y, 2, z, 2, false, 0);
assert(abs(actual - expected) < 1e-12);
end

function test_fastinfo_calccondcmi_matches_reference()
x = int16([0 0 1 1 0 0 1 1]');
y = int16([0 0 1 1 0 1 0 1]');
z = int16([0 0 0 0 1 1 1 1]');
k = int16([0 0 0 0 1 1 1 1]');
[actual, actualK] = fastinfo.calccondcmi(x, 2, y, 2, z, 2, k, 2);
[expected, expectedK] = calccondcmi(x, 2, y, 2, z, 2, k, 2);
assert(abs(actual - expected) < 1e-12);
assert(max(abs(actualK(:) - expectedK(:))) < 1e-12);
end

function test_fastinfo_calcinfo_slice_matches_reference()
x = int16([0 0 1; 0 1 1; 1 0 0; 1 1 0]);
y = int16([0; 0; 1; 1]);
actual = fastinfo.calcinfo_slice(x, 2, y, 2);
expected = zeros(size(actual));
for col = 1:size(x, 2)
    expected(col) = calcinfo(x(:, col), 2, y, 2, false, 0);
end
assert(max(abs(actual(:) - expected(:))) < 1e-12);
end

function test_fastinfo_calccmi_slice_matches_reference()
x = int16([0 0 1; 0 1 1; 1 0 0; 1 1 0]);
y = int16([0; 0; 1; 1]);
z = int16([0; 1; 0; 1]);
actual = fastinfo.calccmi_slice(x, 2, y, 2, z, 2);
expected = calccmi_slice(x, 2, y, 2, z, 2, false, 0);
assert(max(abs(actual(:) - expected(:))) < 1e-12);
end

function test_fastinfo_calcpairwiseinfo_matches_reference()
x = [0.1 0.2 0.3 1.0 1.1 1.2 2.0 2.1 2.2]';
y = int16([0 0 0 1 1 1 2 2 2]');
actual = fastinfo.calcpairwiseinfo(x, 3, y, 3);
expected = calcpairwiseinfo(x, 3, y, 3);
assert(max(abs(actual(:) - expected(:))) < 1e-12);
end

function assert_error_contains(fun, messageFragment)
didError = false;
try
    fun();
catch ME
    didError = true;
    assert(contains(ME.message, messageFragment), ...
        'Unexpected error message: %s', ME.message);
end
if ~didError
    error('Expected error containing "%s" was not raised.', messageFragment);
end
end
