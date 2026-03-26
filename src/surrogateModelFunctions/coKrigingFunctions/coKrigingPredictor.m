function prediction = coKrigingPredictor(xQuery, data, hyperparams, math)
% coKrigingPredictor  Evaluate the AR(1) co-kriging surrogate model.
%
% This function evaluates the posterior mean of the autoregressive co-kriging 
% model at query points using precomputed augmented system vectors.
%
% INPUTS
%   xQuery      : [m x d] Matrix of query points in unit space.
%   data        : Struct containing training data (.xC and .xE required).
%   hyperparams : Struct containing optimized parameters (rho, thetaC, pC, etc.).
%   math        : Struct containing derived augmented matrices (mu, beta, sigma2C, sigma2D).
%
% OUTPUT
%   prediction  : [m x 1] Vector of high-fidelity predictions.
arguments
    xQuery      (:, :) double
    data        (1, 1) struct
    hyperparams (1, 1) struct
    math        (1, 1) struct
end

xC = data.xC;
xE = data.xE;

rho      = hyperparams.rho;

thetaC   = hyperparams.thetaC;
pC       = hyperparams.pC;
polyDegC = math.polyDegreeC;

thetaD   = hyperparams.thetaD;
pD       = hyperparams.pD;
polyDegD = math.polyDegreeD;

% L = math.L;
muC     = math.muC;
muD     = math.muD;
% alpha = math.alpha;
beta    = math.beta;
sigma2C = math.sigma2C;
sigma2D = math.sigma2D;

% Evaluate cross-correlation between xQuery and the augmented dataset [xC; xE]
% cNew is [(nC + nE) x m]
cNew = crossCorrMatrixMF(xQuery, xC, xE, rho, thetaC, pC, sigma2C, thetaD, pD, sigma2D);

% Polynomial query matrices
fCQuery = buildPolynomialMatrix(xQuery, polyDegC);
fDQuery = buildPolynomialMatrix(xQuery, polyDegD);

trend = rho * (fCQuery * muC) + (fDQuery * muD);

% Compute the final prediction
prediction = trend + cNew' * beta;

end

