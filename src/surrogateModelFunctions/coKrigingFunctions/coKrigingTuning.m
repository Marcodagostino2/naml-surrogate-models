function [hyperparams, math] = coKrigingTuning(xC, yC, xE, yE, options)
% coKrigingTuning  Fit an AR(1) Co-Kriging model using Gradient-Enhanced SQP.
%
% Optimizes an autoregressive co-kriging model via a decoupled 2-step process:
%   1) Fit a standard Kriging model on the Low-Fidelity (cheap) dataset.
%   2) Tune the discrepancy model on the High-Fidelity (expensive) dataset.
%      The scaling factor (rho) is profiled analytically via Generalized Least 
%      Squares (GLS) alongside the trend, enabling robust gradient-based tuning.
%
% INPUTS
%   xC      : [nC x d] Matrix of cheap dataset inputs.
%   yC      : [nC x nOut] Matrix of cheap dataset outputs.
%   xE      : [nE x d] Matrix of expensive dataset inputs.
%   yE      : [nE x nOut] Matrix of expensive dataset outputs.
%   options : Struct containing bounds and SQP/Multi-Start settings.
%
% OUTPUTS
%   hyperparams : {1 x nOut} Struct array with optimized parameters
%                 (rho, thetaC, pC, lambdaC, thetaD, pD, lambdaD).
%   math        : {1 x nOut} Struct array with derived augmented matrices.
arguments
    xC      (:, :) double
    yC      (:, :) double
    xE      (:, :) double
    yE      (:, :) double
    options (1, 1) struct
end

d    = size(xC, 2);
nOut = size(yC, 2);

% Extract multi-start settings
numStarts = options.numStarts;

polyDegreeD = options.polyDegreeD;

% Prepare options and bounds for sub-models
cheapOpt  = coDefaultCheapOptions(d, options);
boundsD   = buildCoKrigingBounds(d, options); % Discrepancy bounds (excludes rho)
optimOpts = buildOptimizerOptions(options);

% Build design matrix for polynomial trend
FD = buildPolynomialMatrix(xE, polyDegreeD);

for i = 1:nOut
    yCurrentC = yC(:, i);
    yCurrentE = yE(:, i);
    
    % Tune Low-Fidelity Model
    [hypCell, mathCell] = krigingTuning(xC, yCurrentC, cheapOpt);
    hypC  = hypCell(1);
    mathC = mathCell(1);
    
    % Evaluate Low-Fidelity model at High-Fidelity locations
    dummyData.x = xC;
    yCAtE       = krigingPredictor(xE, dummyData, hypC, mathC);
    
    % Tune Discrepancy Model
    % Note: Rho is profiled analytically inside the NLL function via GLS.
    zStarts = generateMultiStarts(boundsD, numStarts);
    
    plotFun = setupOptimizationPlot(sprintf('Discrepancy Tuning (Out %d)', i), d);
    optimOpts.OutputFcn = plotFun;
    
    bestLoss  = inf;
    bestParam = zStarts(1, :);
    objFun = @(z) coNegLogLikelihood(z, xE, yCurrentE, yCAtE, d, FD);
    
    for s = 1:numStarts
        fprintf('  Disc. Start %d/%d... ', s, numStarts);
        [zOpt, fval] = fmincon(objFun, zStarts(s, :), [], [], [], [], boundsD.lb, boundsD.ub, [], optimOpts);
        fprintf('Loss: %.4f\n', fval);
        
        if fval < bestLoss
            bestLoss = fval;
            bestParam = zOpt;
        end
    end
    
    % Unpack Discrepancy Parameters and Extract Optimal Rho
    [thetaD, pD, lambdaD] = unpackCoKrigingParameters(bestParam, d);
    
    % Call the objective function one last time to retrieve the analytical rho
    [~, ~, rho, ~] = coNegLogLikelihood(bestParam, xE, yCurrentE, yCAtE, d, FD);
    
    % Compute exact discrepancy target and build independent math struct
    yD = yCurrentE - rho * yCAtE;
    mathD = buildKrigingMath(xE, yD, thetaD, pD, lambdaD, polyDegreeD);
    
    hypD.theta  = thetaD;
    hypD.p      = pD;
    hypD.lambda = lambdaD;
    
    % Build Augmented Co-Kriging System
    augMath = buildCoKrigingMath(xC, yCurrentC, xE, yCurrentE, hypC, mathC, hypD, mathD, rho);
    
    hyperStruct.rho     = rho;
    hyperStruct.thetaC  = hypC.theta;
    hyperStruct.pC      = hypC.p;
    hyperStruct.lambdaC = hypC.lambda;
    hyperStruct.thetaD  = thetaD;
    hyperStruct.pD      = pD;
    hyperStruct.lambdaD = lambdaD;
    
    if i == 1
        hyperparams = repmat(hyperStruct, 1, nOut);
        math        = repmat(augMath, 1, nOut);
    else
        hyperparams(i) = hyperStruct;
        math(i)        = augMath;
    end
end
end