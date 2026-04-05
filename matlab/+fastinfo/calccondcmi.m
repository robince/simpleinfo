function [I, IK] = calccondcmi(x, xb, y, yb, z, zb, k, kb, varargin)
p = inputParser;
p.addParameter('Validate', true, @(v) islogical(v) && isscalar(v));
p.parse(varargin{:});
opts = p.Results;

ensure_native_path();
if opts.Validate
    validateattributes(x, {'numeric'}, {'real', 'vector', 'nonempty'});
    validateattributes(y, {'numeric'}, {'real', 'vector', 'nonempty'});
    validateattributes(z, {'numeric'}, {'real', 'vector', 'nonempty'});
    validateattributes(k, {'numeric'}, {'real', 'vector', 'nonempty'});
end

if exist('fastinfo_calccondcmi_cpp', 'file') == 3
    [I, IK] = fastinfo_calccondcmi_cpp(x, double(xb), y, double(yb), z, double(zb), k, double(kb));
else
    [I, IK] = feval('calccondcmi', x, xb, y, yb, z, zb, k, kb);
end
end
