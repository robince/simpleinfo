function x = numdec2base(d, b, m)
%NUMDEC2BASE Convert decimal integers into base-B words.
%   D is interpreted as a row vector of non-negative integers.

validateattributes(d, {'numeric'}, {'real', 'vector', 'nonempty'});
validateattributes(b, {'numeric'}, {'real', 'scalar', 'integer', '>=', 2});

d = double(reshape(d, 1, []));
if any(d < 0) || any(mod(d, 1) ~= 0)
    error('NUMDEC2BASE:InvalidInput', 'D must contain non-negative integers.');
end

requiredDigits = iRequiredBaseDigits(max(d), b);
if nargin < 3 || isempty(m)
    m = requiredDigits;
end
validateattributes(m, {'numeric'}, {'real', 'scalar', 'integer', 'positive'});
if m < requiredDigits
    error('NUMDEC2BASE:WidthTooSmall', ...
        'M is too small to represent the largest value in D.');
end

powers = b .^ ((m - 1):-1:0)';
x = floor(rem(d, b .* powers) ./ powers);
end

function digits = iRequiredBaseDigits(maxValue, base)
digits = 1;
while maxValue >= base
    maxValue = floor(maxValue / base);
    digits = digits + 1;
end
end
