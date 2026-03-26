function xUnit = real2unit(xReal, ranges, power)
% REAL2UNIT  Maps real-domain samples to normalized space [0,1]^k.
%            Inverse of unit2real(), supporting 'uniform' and 'power law'.
%
%   INPUT:
%       xReal    : (N×k) matrix of physical samples
%       ranges   : (2×k) matrix of [lower, upper] bounds for each dimension
%       power    : (1×k) vector of power-law exponents (1 = uniform mapping)
%
%   OUTPUT:
%       xUnit    : (N×k) matrix normalized to [0,1]^k

k = size(ranges, 2);

% Check coherence
if size(xReal, 2) ~= k
    error('xReal must have one column per dimension (expected %d, got %d).', k, size(xReal,2));
end

% Extract bounds and mapping
rmin = ranges(1, :);
rmax = ranges(2, :);

% Linear de-normalization (inverse of scaling)
U = (xReal - rmin) ./ (rmax - rmin);

% Inverse power-law transform
U = U.^ (1 ./ power);

xUnit = U;

end
