function [yPred, varPred] = coKrigingPredictor(xQuery, data, hyperparams, math)
% coKrigingPredictor  Evaluate the Recursive Co-Kriging surrogate model.
%
% This function evaluates the predictive mean and predictive variance of the 
% decoupled AR(1) Co-Kriging model (Le Gratiet 2013) at query points.
% It heavily exploits precomputed vectors from buildCoKrigingMath to ensure
% execution in O(m * nE) time, without rebuilding joint covariance matrices.
%
% INPUTS
%   xQuery      : [m x d] Matrix of query points in unit space.
%   data        : Struct containing training data (.xC and .xE required).
%   hyperparams : Struct containing optimized parameters (betaRho, thetaD, etc.).
%   math        : Struct containing derived precomputed matrices.
%
% OUTPUTS
%   yPred       : [m x 1] Vector of high-fidelity predictions (Mean).
%   varPred     : [m x 1] Vector of high-fidelity predictive variances.

arguments
    xQuery      (:, :) double
    data        (1, 1) struct
    hyperparams (1, 1) struct
    math        (1, 1) struct
end

% Extract parameters
pD = math.polyDegreeD;
pRho = math.polyDegreeRho;

% Evaluate the Low-Fidelity Kriging model at the query points
dummyData.x = data.xC;

hyperparamsC.theta  = hyperparams.thetaC;
hyperparamsC.p      = hyperparams.pC;
hyperparamsC.lambda = hyperparams.lambdaC;

[yCPred, varCPred] = krigingPredictor(xQuery, dummyData, hyperparamsC, math.mathC);

% Build polynomial bases at the query locations
fDQuery   = buildPolynomialMatrix(xQuery, pD);
fRhoQuery = buildPolynomialMatrix(xQuery, pRho);

% Construct the augmented design matrix h(x) for the query points
% hQuery = [fD(x), rhoBasis(x) * yC(x)]
hQuery = [fDQuery, fRhoQuery .* yCPred];

% Evaluate spatial cross-correlation r_D(x) between xE and xQuery
% rD has dimension [nE x m]
rD = crossCorrMatrix(math.xE, xQuery, hyperparams.thetaD, hyperparams.pD);

% -------------------------------------------------------------------------
% PREDICTIVE MEAN (Le Gratiet 2013, Eq. 17)
% -------------------------------------------------------------------------
% The term (hQuery * math.lambdaHat) expands to:
% fDQuery * muD + fRhoQuery * betaRho .* yCPred
% This intrinsically computes both the discrepancy trend and rho(x)*yCPred!
trendMF = hQuery * math.lambdaHat;

% Add the stochastic innovation: r_D(x)^T * R_D^-1 * (yE - H * lambdaHat)
yPred = trendMF + rD' * math.betaD;

% -------------------------------------------------------------------------
% PREDICTIVE VARIANCE (Le Gratiet 2013, Eq. 18)
% -------------------------------------------------------------------------
if nargout > 1
    % Component transported from the Low-Fidelity model: rho(x)^2 * varC(x)
    % Extract betaRho (the last rhoBasis elements of lambdaHat)
    rhoBasis = size(fRhoQuery, 2);

    betaRho = math.lambdaHat(end - rhoBasis + 1: end);
    rhoPred = fRhoQuery * betaRho;
    varRhoC = (rhoPred .^ 2) .* varCPred;
    
    % Base Kriging spatial variance: 1 - r_D(x)^T * R_D^-1 * r_D(x)
    invRrD = math.lD' \ (math.lD \ rD); % [nE x m]
    varGP  = 1 - sum(rD .* invRrD, 1)';   % Fast vectorized diagonal, [m x 1]
    varGP(varGP < 0) = 0;                 % Numerical safety cap
    
    % Universal Kriging inflation due to GLS parameter uncertainty:
    % u(x)^T * Sigma_t * u(x), where u(x) = h(x) - H^T * R_D^-1 * r_D(x)
    % math.RinvH is precomputed (R_D^-1 * H)
    uQuery = hQuery - (math.rInvH' * rD)'; % [m x (pD + pRho)]
    varTrend = sum((uQuery * math.covBeta) .* uQuery, 2); % Vectorized diag, [m x 1]
    
    % Total Discrepancy Variance
    varDPred = math.sigma2D * (varGP + varTrend);
    
    % Final Multi-Fidelity Predictive Variance
    varPred = varRhoC + varDPred;
end

end