function bounds = buildKrigingBounds(d, options)
% buildKrigingBounds  Assemble fmincon lower/upper bounds for Kriging hyperparameters.
%
% Decision vector layout:
%   z = [log10(theta_1..theta_d), p_1..p_d, lambda]
%
% INPUTS
%   d       : number of input dimensions.
%   options : struct with (possibly empty) fields:
%       - lbLogTheta, ubLogTheta : [1 x d] bounds for log10(theta)
%       - lbP, ubP               : [1 x d] bounds for p
%       - lbLambda, ubLambda     : scalar bounds for lambda
%
% OUTPUT
%   bounds : struct with fields lb, ub, nVars.
arguments
    d (1,1) {mustBeInteger, mustBePositive}
    options (1,1) struct
end

lbLogTheta = defaultIfEmpty(options.lbLogTheta, -5 * ones(1, d));
ubLogTheta = defaultIfEmpty(options.ubLogTheta,  2 * ones(1, d));
lbP        = defaultIfEmpty(options.lbP,         1 * ones(1, d));
ubP        = defaultIfEmpty(options.ubP,         2 * ones(1, d));

lbLambda = defaultIfEmpty(options.lbLambda, 0);
ubLambda = defaultIfEmpty(options.ubLambda, 0);
% Minimal shape checks to avoid silent dimension mismatches later.
assert(numel(lbLogTheta) == d && numel(ubLogTheta) == d, "lb/ubLogTheta must have length d.");
assert(numel(lbP) == d && numel(ubP) == d, "lb/ubP must have length d.");

bounds.lb    = [lbLogTheta(:).', lbP(:).', lbLambda];
bounds.ub    = [ubLogTheta(:).', ubP(:).', ubLambda];
bounds.nVars = numel(bounds.lb);

end
