function [hyperparams, math] = krigingTuning(x, y, options)
% krigingTuning  Fit a Kriging surrogate model using Gradient-Enhanced SQP.
%
% Optimizes the Kriging hyperparameters for each output dimension independently.
% The optimization leverages analytical gradients and a multi-start Sequential
% Quadratic Programming (SQP) algorithm to minimize the Concentrated Negative 
% Log-Likelihood (NLL). Can automatically switch to Gradient-Enhanced Kriging 
% (GEK) if gradient data is provided in the options struct.
%
% INPUTS
%   x       : [n x d] Matrix of unit-space input coordinates.
%   y       : [n x nOut] Matrix of target output values.
%   options : Struct containing tuning configurations (bounds, SQP settings,
%             and optionally 'dy' and 'dyDirs' for GEK).
%
% OUTPUTS
%   hyperparams : {1 x nOut} Struct array. Each cell contains optimized parameters.
%   math        : {1 x nOut} Struct array. Each cell contains derived matrices.
arguments
    x       (:, :) double
    y       (:, :) double
    options (1, 1) struct
end

d      = size(x, 2);
nOut   = size(y, 2);

% Check if Gradient-Enhanced Kriging (GEK) is activated
useGEK = isfield(options, 'dy') && isfield(options, 'dyDirs') && ~isempty(options.dy);

% Assemble bounds and optimizer options
numStarts  = options.numStarts;
bounds     = buildKrigingBounds(d, options);
optimOpts  = buildOptimizerOptions(options);
polyDegree = options.polyDegree;

for i = 1:nOut
    yCurrent = y(:, i);
    
    % Generate multi-start initial points within bounds
    zStarts = generateMultiStarts(bounds, numStarts);
    
    % Setup closure for live optimization plotting
    plotFun = setupOptimizationPlot(sprintf('Kriging Tuning (Out %d)', i), d);
    optimOpts.OutputFcn = plotFun;
    
    bestLoss  = inf;
    bestParam = zStarts(1, :);
    
    % Build standard design matrix for polynomial trend
    F = buildPolynomialMatrix(x, polyDegree);
    
    % Branch Objective Function wrapper (GEK vs Standard)
    if useGEK
        dyCurrent = options.dy;
        dyDirs    = options.dyDirs;
        optimOpts.SpecifyObjectiveGradient = false;
        objFun = @(z) negLogLikelihoodGEK(z, x, yCurrent, dyCurrent, dyDirs, d, F, polyDegree);
    else
        dyCurrent = [];
        dyDirs    = [];
        objFun = @(z) negLogLikelihood(z, x, yCurrent, d, F);
    end
    
    % Multi-start optimization loop
    for s = 1:numStarts
        fprintf('  Start %d/%d... ', s, numStarts);
        [zOpt, fval] = fmincon(objFun, zStarts(s, :), [], [], [], [], bounds.lb, bounds.ub, [], optimOpts);
        fprintf('Loss: %.4f\n', fval);
        
        % Keep track of the global minimum
        if fval < bestLoss
            bestLoss = fval;
            bestParam = zOpt;
        end
    end
    
    % Extract optimal parameters from decision vector
    [theta, p, lambda] = unpackKrigingParameters(bestParam, d);
    
    % Branch automatically mathematical components builder (GEK vs Standard)
    mathStruct = buildKrigingMath(x, yCurrent, theta, p, lambda, polyDegree, dyCurrent, dyDirs);
    
    % Package hyperparameter struct
    hyperStruct.theta  = theta;
    hyperStruct.p      = p;
    hyperStruct.lambda = lambda;
    
    % Assign to output arrays
    if i == 1
        hyperparams = repmat(hyperStruct, 1, nOut);
        math        = repmat(mathStruct, 1, nOut);
    else
        hyperparams(i) = hyperStruct;
        math(i)        = mathStruct;
    end
end
end
