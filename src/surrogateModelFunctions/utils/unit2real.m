function xReal = unit2real(xUnit, ranges, power)
% UNIT2REAL  Maps normalized samples X ∈ [0,1]^k to their real-domain equivalents,
%             supporting both 'uniform' and 'power law' mapping types.
%
%   INPUT:
%       xUnit     : (N×k) matrix of adimensional samples (rows = points, cols = dims)
%       ranges    : (2xk) matrix of [lower, upper] bounds for each dimension
%       power     : (1xk) vector of power-law exponents (1 = uniform mapping)
%
%   OUTPUT:
%       xReal     : (N×k) matrix of samples mapped into physical space
%
%   NOTE:
%       - Fully vectorized (no for-loops over rows).
%       - Requires `fltcon.dimNames` and mapping info.

k = size(ranges, 2);

% Check coherence
if size(xUnit, 2) ~= k
    error('xUnit must have one column per dimension (expected %d, got %d).', k, size(xUnit,2));
end

% Extract lower/upper bounds

rmin = ranges(1, :);
rmax = ranges(2, :);

% Base normalized vector
U = xUnit;

% Apply power-law transform 
U = U .^ power;

% Linear scaling (works for both, because power law handled above)
xReal = rmin + U .* (rmax - rmin);
end
