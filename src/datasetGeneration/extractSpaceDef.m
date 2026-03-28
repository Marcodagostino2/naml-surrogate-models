function [ranges, power] = extractSpaceDef(opts, activeNames)
% extractSpaceDef  Extract lower/upper bounds and power mapping from config.
%
% Used by both coeffsGridInterp (unit→real) and coeffsCsvInterp (real→unit).
% The caller decides the conversion direction via unit2real / real2unit.
lb    = arrayfun(@(d) opts.(d).lower, activeNames);
ub    = arrayfun(@(d) opts.(d).upper, activeNames);
power = arrayfun(@(d) opts.(d).power, activeNames);
ranges = [lb; ub];
end
