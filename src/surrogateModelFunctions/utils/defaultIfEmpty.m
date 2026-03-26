function out = defaultIfEmpty(value, default)
%DEFAULTIFEMPTY Return default if value is empty.

if isempty(value)
    out = default;
else
    out = value;
end
end
