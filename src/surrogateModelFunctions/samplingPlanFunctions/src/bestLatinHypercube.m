function [bestLh,qBest,bestPhi] = bestLatinHypercube(nPoints, nDim, ...
    population, maxIterations, metric, qPhiSort, sortCriterion, qOptimization)
% Generates an optimized Latin hypercube by optimizing the Morris-Mitchell
% criterion for a range of exponents and plots the first two dimensions of
% the current hypercube throughout the optimization process.
%
% Inputs:
%       nPoints - number of points required
%       nDim - number of design variables
%       Population - number of individuals in the evolutionary operation
%                    optimizer
%       Iterations - number of generations the evolutionary operation
%                    optimizer is run for
%       Note: high values for the two inputs above will ensure high quality
%       hypercubes, but the search will take longer.
%
% Outputs:
%       bestLh - optimized Latin hypercube

if nDim < 2
	error('Latin hypercubes are not defined for nDim < 2');
end

% List of qs to optimize Phi_q for
q = qOptimization;

% Multi-start initialization
nInitialStarts = max(3, round(population/4));
xCand = zeros(nPoints, nDim, nInitialStarts);
phiP  = zeros(1, nInitialStarts);

for j = 1: nInitialStarts
    xCand(:,:,j) = randLatinHypercube(nPoints, nDim, 1);
    phiP(j)      = maxminPhi(xCand(:,:,j), metric, qOptimization(1));
end

[~, idx0] = min(phiP);
xStart = xCand(:,:,idx0);

fprintf('[INFO] Best among %d initial LHS used as starting point.\n', ...
    nInitialStarts);

x3D    = nan(nPoints, nDim, length(q));
phi    = nan(1, length(q));
% For each q optimize Phi_q
for i = 1: length(q)
    fprintf('Now optimizing for q=%d...\n',q(i));
    [x3D(:,:,i), phi(i)] = phiOptimizer(xStart, population, ...
        maxIterations, metric, q(i));
end

switch sortCriterion
    case 'maxmin'
        % Sort according to the Morris-Mitchell criterion
        index = maxminSort(x3D ,metric);
    case 'phi_q'
        index = phiSort(x3D, metric, qPhiSort);
end

fprintf('Best lh found using q=%d...\n', q(index(1)));

% And the Latin hypercube with the best space-filling properties is...
bestLh  = x3D(:,:,index(1));
bestPhi = phi(index(1));

qBest = q(index(1));