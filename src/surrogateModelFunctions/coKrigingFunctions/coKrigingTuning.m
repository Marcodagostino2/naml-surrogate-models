function [hyperparams, math] = coKrigingTuning(xC, yC, xE, yE, options)
% coKrigingTuning  Fit a Recursive Co-Kriging model using Gradient-Enhanced SQP.
%
% Optimizes an autoregressive co-kriging model via a decoupled 2-step process
% following Le Gratiet's formulation:
%   1) Fit a standard Kriging model on the Low-Fidelity (cheap) dataset.
%   2) Tune the discrepancy model on the High-Fidelity (expensive) dataset.
%      The spatially variable scaling factor rho(x) = fRho * betaRho is 
%      profiled analytically via Generalized Least Squares (GLS).
%
% INPUTS
%   xC      : [nC x d] Matrix of cheap dataset inputs.
%   yC      : [nC x nOut] Matrix of cheap dataset outputs.
%   xE      : [nE x d] Matrix of expensive dataset inputs.
%   yE      : [nE x nOut] Matrix of expensive dataset outputs.
%   options : Struct containing bounds, polynomial degrees, and SQP settings.
%
% OUTPUTS
%   hyperparams : {1 x nOut} Struct array with optimized parameters
%                 (betaRho, thetaC, pC, lambdaC, thetaD, pD, lambdaD).
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

% Extract multi-start and trend settings
numStarts   = options.numStarts;
polyDegreeD   = options.polyDegreeD;
polyDegreeRho = options.polyDegreeRho;

% Prepare options and bounds for sub-models
cheapOpt  = coDefaultCheapOptions(d, options);
boundsD   = buildCoKrigingBounds(d, options); % Discrepancy bounds (excludes rho)
optimOpts = buildOptimizerOptions(options);

% Build design matrices for polynomial trends (evaluated at expensive points)
fD   = buildPolynomialMatrix(xE, polyDegreeD);
fRho = buildPolynomialMatrix(xE, polyDegreeRho);

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
    % Note: betaRho is profiled analytically inside the NLL function via GLS.
    zStarts = generateMultiStarts(boundsD, numStarts);
    
    plotFun = setupOptimizationPlot(sprintf('Discrepancy Tuning (Out %d)', i), d);
    optimOpts.OutputFcn = plotFun;
    
    bestLoss  = inf;
    bestParam = zStarts(1, :);
    objFun = @(z) coNegLogLikelihood(z, xE, yCurrentE, yCAtE, d, fD, fRho);
    
    for s = 1:numStarts
        fprintf('  Disc. Start %d/%d... ', s, numStarts);
        try
            [zOpt, fval] = fmincon(objFun, zStarts(s, :), [], [], [], [], boundsD.lb, boundsD.ub, [], optimOpts);
            fprintf('Loss: %.4f\n', fval);
            
            if fval < bestLoss
                bestLoss = fval;
                bestParam = zOpt;
            end
        catch
            fprintf('Factorization failed.\n');
            continue;
        end
    end
    
    % Unpack Discrepancy Parameters
    [thetaD, pD, lambdaD] = unpackCoKrigingParameters(bestParam, d);
    
    % Call the objective function one last time to retrieve the analytical betaRho
    [~, ~, betaRho, ~] = coNegLogLikelihood(bestParam, xE, yCurrentE, yCAtE, d, fD, fRho);
    
    % Compute exact discrepancy target by subtracting the spatially scaled cheap prediction
    % yD(x) = yE(x) - [fRho(x) * betaRho] * yCAtE(x)
    yD = yCurrentE - (fRho * betaRho) .* yCAtE;
    
    % Build independent math struct for the discrepancy process
    mathD = buildKrigingMath(xE, yD, thetaD, pD, lambdaD, polyDegreeD);
    
    hypD.theta  = thetaD;
    hypD.p      = pD;
    hypD.lambda = lambdaD;
    
    % Build Augmented Co-Kriging System 
    augMath = buildCoKrigingMath(xC, xE, yCurrentE, hypC, mathC, hypD, mathD, polyDegreeRho);
    
    hyperStruct.betaRho = betaRho;
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