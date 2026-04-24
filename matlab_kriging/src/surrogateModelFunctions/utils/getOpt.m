function v = getOpt(s, name, default)
if isfield(s,name) && ~isempty(s.(name)), v = s.(name); else, v = default; end
end