function out = sanitizeFilename(name)
out = string(regexprep(char(name), '[^\w\-]', '_'));
end
