function validate_native_integer_class(values, name)
if ~(isa(values, 'int16') || isa(values, 'int32') || isa(values, 'int64'))
    error('fastinfo:type', ...
        '%s must have class int16, int32, or int64 for the native fast path.', ...
        name);
end
end
