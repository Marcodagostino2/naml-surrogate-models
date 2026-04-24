function cNew = crossCorrMatrixMF(xQuery, xC, xE, rho, thetaC, pC, sigma2C, thetaD, pD, sigma2D)
%crossCorrMatrixMF  Build the augmented cross-covariance matrix for co-kriging.
%
% Computes the cross-covariance between the m query points and the augmented 
% training set (cheap and expensive samples) based on the AR(1) formulation:
%   y_H(x) = rho * y_L(x) + delta(x)
%
% INPUTS
%   xQuery  : [m x d] matrix of m query points.
%   xC, xE  : [nC x d], [nE x d] training locations for LF and HF data.
%   rho     : [1 x 1] scaling factor mapping LF to HF.
%   thetaC, pC, sigma2C : Hyperparameters for the cheap (LF) GP.
%   thetaD, pD, sigma2D : Hyperparameters for the discrepancy GP.
%
% OUTPUT
%   cNew : [(nC + nE) x m] stacked augmented cross-covariance matrix.

% Covariance with cheap points: Cov(y_H(xQuery), y_L(xC))
cC = rho * sigma2C * crossCorrMatrix(xC, xQuery, thetaC, pC);

% Covariance with expensive points: Cov(y_H(xQuery), y_H(xE))
cE = (rho^2) * sigma2C * crossCorrMatrix(xE, xQuery, thetaC, pC) + ...
      sigma2D * crossCorrMatrix(xE, xQuery, thetaD, pD);

% Stack to match the augmented vector layout yAug = [yC; yE]
cNew = [cC; cE];

end