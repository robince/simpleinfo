function d = numbase2dec(x, b)
%NUMBASE2DEC Convert base-B words into decimal integers.
%   X is M x Nt, representing Nt words of length M using digits 0:(B-1).

validateattributes(x, {'numeric'}, {'real', '2d', 'nonempty'});
validateattributes(b, {'numeric'}, {'real', 'scalar', 'integer', '>=', 2});

if any(x(:) < 0) || any(mod(x(:), 1) ~= 0) || any(x(:) >= b)
    error('NUMBASE2DEC:InvalidDigits', 'X must contain integer digits in the range 0:(B-1).');
end

m = size(x, 1);
powers = b .^ ((m - 1):-1:0);
d = powers * double(x);
end
