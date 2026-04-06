function results = run_fastinfo_cpp_mex_smoke_tests()
results = struct('passed', 0, 'failed', 0, 'details', {{}});
tests = { ...
    @test_eqpop_scalar, ...
    @test_eqpop_tied_values_error, ...
    @test_eqpop_sorted_scalar, ...
    @test_eqpop_slice, ...
    @test_eqpop_sorted_slice, ...
    @test_calcinfo_scalar, ...
    @test_calcinfomatched, ...
    @test_calccmi_scalar, ...
    @test_calccondcmi, ...
    @test_calccmi_slice, ...
    @test_calcinfo_slice, ...
    @test_calcinfoperm, ...
    @test_calcinfoperm_slice};
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
fprintf('fastinfo_cpp_mex test summary: %d passed, %d failed\n', results.passed, results.failed);
if results.failed > 0
    error('fastinfo_cpp_mex_test:failuresDetected', 'fastinfo_cpp_mex smoke tests failed.');
end
end

function test_eqpop_scalar()
x = [10 0 20 30 11 1 21 31]';
xb = fastinfo_eqpop_cpp(x, 4);
assert(isequal(double(xb(:))', [1 0 2 3 1 0 2 3]));
end

function test_eqpop_tied_values_error()
x = [0 0 0 1 1 1]';
assert_error_contains(@() fastinfo_eqpop_cpp(x, 4), ...
    'Cannot form the requested number of equal-population bins');
end

function test_eqpop_sorted_scalar()
x = [0 1 2 3 4 5 6 7]';
xb = fastinfo_eqpop_sorted_cpp(x, 4);
assert(isequal(double(xb(:))', [0 0 1 1 2 2 3 3]));
end

function test_eqpop_slice()
x = [10 0 0; 0 1 0; 20 2 0; 30 3 1; 11 4 1; 1 5 1; 21 6 1; 31 7 1]';
x = x';
xb = fastinfo_eqpop_slice_cpp(x, 4, 2);
assert(isequal(xb(:, 1), [1; 0; 2; 3; 1; 0; 2; 3]));
assert(isequal(xb(:, 2), [0; 0; 1; 1; 2; 2; 3; 3]));
assert(all(isnan(xb(:, 3))));
end

function test_eqpop_sorted_slice()
x = [0 0 0; 0 1 0; 1 2 0; 1 3 1; 2 4 1; 2 5 1; 3 6 1; 3 7 1];
xb = fastinfo_eqpop_sorted_slice_cpp(x, 4, 2);
assert(isequal(xb(:, 1), [0; 0; 1; 1; 2; 2; 3; 3]));
assert(isequal(xb(:, 2), [0; 0; 1; 1; 2; 2; 3; 3]));
assert(all(isnan(xb(:, 3))));
end

function test_calcinfo_scalar()
x = int16([0 0 1 1]');
y = int16([0 0 1 1]');
actual = fastinfo_calcinfo_cpp(x, 2, y, 2);
assert(abs(actual - 1.0) < 1e-12);
end

function test_calccmi_scalar()
x = int16([0 0 1 1 0 0 1 1]');
y = int16([0 0 1 1 0 1 0 1]');
z = int16([0 0 0 0 1 1 1 1]');
actual = fastinfo_calccmi_cpp(x, 2, y, 2, z, 2);
assert(abs(actual - 0.5) < 1e-12);
end

function test_calcinfomatched()
x = int16([0 0 1; 0 1 1; 1 0 0; 1 1 0]);
y = int16([0 1 1; 0 0 1; 1 1 0; 1 0 0]);
actual = fastinfo_calcinfomatched_cpp(x, 2, y, 2, 2);
expected = [1; 1; 1];
assert(max(abs(actual(:) - expected(:))) < 1e-12);
end

function test_calccondcmi()
x = int16([0 0 1 1 0 1 1 0 0 0 1 1 0 1 1 0]');
y = int16([0 0 1 1 0 1 1 0 0 1 0 1 0 1 1 0]');
z = int16([0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1]');
k = int16([0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1]');
[actual, contributions] = fastinfo_calccondcmi_cpp(x, 2, y, 2, z, 2, k, 2);
[expected, expectedContributions] = calccondcmi(x, 2, y, 2, z, 2, k, 2);
assert(abs(actual - expected) < 1e-12);
assert(max(abs(contributions(:) - expectedContributions(:))) < 1e-12);
end

function test_calccmi_slice()
x = int16([0 0 1; 0 1 1; 1 0 0; 1 1 0]);
y = int16([0; 0; 1; 1]);
z = int16([0; 1; 0; 1]);
actual = fastinfo_calccmi_slice_cpp(x, 2, y, 2, z, 2, 2);
expected = [1; 0; 1];
assert(max(abs(actual(:) - expected(:))) < 1e-12);
end

function test_calcinfo_slice()
x = int16([0 0 1; 0 1 1; 1 0 0; 1 1 0]);
y = int16([0; 0; 1; 1]);
actual = fastinfo_calcinfo_slice_cpp(x, 2, y, 2, 2);
expected = [1; 0; 1];
assert(max(abs(actual(:) - expected(:))) < 1e-12);
end

function test_calcinfoperm()
x = int16(repmat([0; 1], 128, 1));
y = int16(repmat([0; 1], 128, 1));
actual = fastinfo_calcinfoperm_cpp(x, 2, y, 2, 64, 2, 123);
reference = permutation_reference_scalar(x, 2, y, 2, 64, 123);
assert(isvector(actual) && numel(actual) == 64);
assert(all(isfinite(actual)));
assert(all(actual >= -1e-12));
assert(mean(actual) < 0.02);
assert_reference_distribution_close(actual, reference);
end

function test_calcinfoperm_slice()
x = int16(repmat([0 0 1; 0 1 1; 1 0 0; 1 1 0], 32, 1));
y = int16(repmat([0; 1; 0; 1], 32, 1));
nperm = 24;
a = fastinfo_calcinfoperm_slice_cpp(x, 2, y, 2, nperm, 2, 123);
b = fastinfo_calcinfoperm_slice_cpp(x, 2, y, 2, nperm, 2, 123);
reference = permutation_reference_slice_native(x, 2, y, 2, nperm, 123);
assert(isequal(size(a), [nperm 3]));
assert(all(isfinite(a), 'all'));
assert(max(abs(a(:) - b(:))) < 1e-12);
assert(max(abs(a(:) - reference(:))) < 1e-12);
end

function reference = permutation_reference_scalar(x, xb, y, yb, nperm, seed)
s = rng;
cleanup = onCleanup(@() rng(s));
rng(double(seed), 'twister');
reference = calcinfoperm(x, xb, y, yb, nperm, false, 0);
clear cleanup
end

function reference = permutation_reference_slice(x, xb, y, yb, nperm, seed)
s = rng;
cleanup = onCleanup(@() rng(s));
rng(double(seed), 'twister');
reference = calcinfoperm_slice(x, xb, y, yb, nperm, false, 0);
clear cleanup
end

function reference = permutation_reference_slice_native(x, xb, y, yb, nperm, seed)
reference = zeros(nperm, size(x, 2));
for col = 1:size(x, 2)
    reference(:, col) = fastinfo_calcinfoperm_slice_cpp(x(:, col), xb, y, yb, nperm, 1, double(seed + col - 1));
end
end

function assert_reference_distribution_close(actual, expected)
assert(isequal(size(actual), size(expected)));
assert(all(isfinite(actual(:))));
assert(all(isfinite(expected(:))));
meanTol = 0.03;
stdTol = 0.04;
assert(abs(mean(actual(:)) - mean(expected(:))) < meanTol, ...
    'Permutation mean mismatch: actual=%g expected=%g', mean(actual(:)), mean(expected(:)));
assert(abs(std(actual(:), 1) - std(expected(:), 1)) < stdTol, ...
    'Permutation std mismatch: actual=%g expected=%g', std(actual(:), 1), std(expected(:), 1));
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
