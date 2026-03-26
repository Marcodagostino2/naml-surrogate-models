function zStarts = generateMultiStarts(bounds, numStarts)
% generateMultiStarts  Generate initial points for multi-start optimization.
%
% This function creates a set of initial parameter vectors distributed 
% within the defined lower and upper bounds. The first start is always 
% the exact midpoint of the domain, while subsequent starts are uniformly 
% sampled to ensure robust exploration of the non-convex landscape.
%
% INPUTS
%   bounds    : Struct containing fields lb (lower bounds), ub (upper bounds), 
%               and nVars (number of variables).
%   numStarts : Integer specifying the number of starting points.
%
% OUTPUT
%   zStarts   : [numStarts x nVars] Matrix of initial decision vectors.

    zStarts = zeros(numStarts, bounds.nVars);
    zStarts(1, :) = bounds.lb + 0.5 * (bounds.ub - bounds.lb); % Midpoint start
    if numStarts > 1
        zStarts(2:end, :) = bounds.lb + rand(numStarts-1, bounds.nVars) .* (bounds.ub - bounds.lb);
    end
end