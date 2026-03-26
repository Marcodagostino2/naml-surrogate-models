function [roundedRealSpan, roundedUnitSpan] = createSpan(fltcon)
% createSpan  Build 1D discretization vectors for each flight-condition dimension.
%
% The discretization is defined by fltcon(i).n points in unit space,
% then mapped to the physical domain [lower, upper] according to a mapping rule.
% Rounding is applied to the physical values (roundedRealSpan) and the corresponding
% unit coordinates after rounding are returned (roundedUnitSpan).
%
% Mapping rules:
%   - "uniform"   : real = u*(upper-lower) + lower -> power = 1
%   - "power law" : real = (u^power)*(upper-lower) + lower
%
% Rounding (fltcon(i).round):
%   roundedRealSpan is obtained with round(realSpan, Ndecimals).
%
% INPUT
%   fltcon : Array of structs (length k), each containing:
%           .name, .n, .lower, .upper, .power, .round
%
% OUTPUTS
%   roundedRealSpan : [k x maxN] rounded physical values actually used downstream (zero-padded).
%   roundedUnitSpan : [k x maxN] unit coordinates consistent with roundedRealSpan (zero-padded).
%
% Notes
%   - Rows correspond to fltcon.name. Only the first n values of each row
%     are meaningful; remaining entries are padding.
%   - If n == 1, linspace(0,1,1) returns 1 (single endpoint).
%   - If upper == lower, n must be 1 (constant dimension); otherwise mapping is ill-defined.

% Check dimension consistency and extract parameters
k = numel(fltcon);

nVec = [fltcon.n]; 
maxN = max(nVec);

realSpan        = zeros(k, maxN);
roundedRealSpan = zeros(k, maxN);
roundedUnitSpan = zeros(k, maxN);

for i = 1:k
    n     = fltcon(i).n;
    d     = fltcon(i).name;                          % Extracted for clear error messages
    u     = linspace(0, 1, n);                       % unit grid for this dimension
    lo    = fltcon(i).lower;
    up    = fltcon(i).upper;
    
    if up == lo && n ~= 1
        error("createSpan:InvalidBounds", ...
            "Dimension '%s' has upper == lower but n ~= 1.", d);
    end
    
    power = fltcon(i).power;
    realSpan(i,1:n) = (u.^power)*(up - lo) + lo;
    roundedRealSpan(i,1:n) = round(realSpan(i,1:n), fltcon(i).round);
    
    if up ~= lo
        % Map rounded real values back to the unit coordinate used by the power-law parametrization.
        z = (roundedRealSpan(i,1:n) - lo) / (up - lo);
        roundedUnitSpan(i,1:n) = z.^(1/power);
    else
        roundedUnitSpan(i,1:n) = u;
    end
end
end
