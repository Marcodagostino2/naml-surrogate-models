function math = buildCoKrigingMath(xC, xE, yE, hypC, mathC, hypD, mathD, polyDegreeRho)
% buildCoKrigingMath  Precompute math structures for Recursive Co-Kriging prediction.
%
% Implements the Universal Co-Kriging equations of Le Gratiet (2013), Prop. 2.
% All matrix-vector products needed for fast evaluation of the predictive mean 
% (Eq. 17) and the predictive variance (Eq. 18) are precomputed here and stored 
% in the math struct, avoiding repeated expensive inversions during inference.
%
% INPUTS
%   xC              : [nC x d] LF training inputs.
%   xE, yE          : [nE x d], [nE x 1] HF training inputs and outputs.
%   hypC, mathC     : Tuned hyperparameters and math struct for the LF model.
%   hypD, mathD     : Tuned hyperparameters and math struct for the Discrepancy model.
%   polyDegreeRho   : Degree of the spatial polynomial for rho(x).
%
% OUTPUT
%   math : Struct containing precomputed quantities for fast inference:
%          .mathC       - LF model math struct (unchanged)
%          .xC, .xE     - Training data locations
%          .lD          - Cholesky factor of rD for fast solves
%          .rInvH       - rD^-1 * hT (needed for both mean and variance)
%          .covBeta     - Sigma_t = (H^T rD^-1 H)^-1 (parameter covariance)
%          .lambdaHat   - Sigma_t * nu_t = GLS estimator [muD; betaRho]
%          .betaD       - rD^-1 * (yE - H * lambdaHat) (innovation for mean)
%          .sigma2D     - Hat{Q}_t / (2*a_t): REML variance estimate
%          .polyDegreeD, .polyDegreeRho  - Polynomial degrees

arguments
    xC            (:, :) double
    xE            (:, :) double
    yE            (:, 1) double
    hypC          (1, 1) struct
    mathC         (1, 1) struct
    hypD          (1, 1) struct
    mathD         (1, 1) struct
    polyDegreeRho (1, 1) double
end

nE = size(xE, 1);
d  = size(xE, 2);

% Store independent model and data for cross-correlation during prediction
math.mathC         = mathC;
math.xC            = xC;
math.xE            = xE;
math.polyDegreeD   = mathD.polyDegree;
math.polyDegreeRho = polyDegreeRho;

% --- Build rD: Correlation matrix of the Discrepancy process ---
corrMatrixD = zeros(nE, nE);
for k = 1:d
    corrMatrixD = corrMatrixD + hypD.theta(k) * (abs(xE(:, k) - xE(:, k)') .^ hypD.p(k));
end
rD = exp(-corrMatrixD) + hypD.lambda * eye(nE, class(xE));

[lD, flag] = chol(rD, 'lower');
if flag ~= 0
    error('buildCoKrigingMath:NotPD', ...
        'Discrepancy correlation matrix rD is not positive definite.');
end
math.lD = lD;
solveRD = @(b) (lD' \ (lD \ b));

% --- Build augmented design matrix hT = [fD, diag(yCAtE) * fRho] ---
% This is the Ht matrix used in Eq. (17) and (18) of Le Gratiet.
dummyData.x = xC;
yCAtE = krigingPredictor(xE, dummyData, hypC, mathC);

fD   = buildPolynomialMatrix(xE, mathD.polyDegree);
fRho = buildPolynomialMatrix(xE, polyDegreeRho);
hT   = [fD, fRho .* yCAtE];

% --- Precompute rD^-1 * hT  (used in both mean Eq.17 and variance Eq.18) ---
rInvH        = solveRD(hT);
math.rInvH   = rInvH;

% --- Precompute Sigma_t = (H' * rD^-1 * H)^-1: parameter covariance ---
% (Eq. 15, ninfo case: Sigma_t = [H^T R^-1 H]^-1)
covBeta        = (hT' * rInvH) \ eye(size(hT, 2));
math.covBeta   = covBeta;

% --- Precompute lambda_hat = Sigma_t * nu_t = GLS estimator ---
% nu_t = H' * rD^-1 * yE   (Eq. 15, ninfo)
% lambda_hat = covBeta * nu_t  = [muD; betaRho]  (Eq. 17)
nuBeta           = hT' * solveRD(yE);
lambdaHat        = covBeta * nuBeta;
math.lambdaHat   = lambdaHat; 

% --- Precompute beta_D = rD^-1 * (yE - H * lambda_hat) ---
% This is the innovation term r_t(x)^T * rD^-1 * (...) in Eq. (17)
math.betaD = solveRD(yE - hT * lambdaHat);

% --- Precompute sigma2_D = Q_hat_t / (2 * a_t): REML variance estimate ---
% Q_hat_t = (yE - H * lambdaHat)' * rD^-1 * (yE - H * lambdaHat)  (Eq. 16)
% a_t = (nE - pD - pRho) / 2  (ninfo case, Eq. 16)
pD   = size(fD, 2);
pRho = size(fRho, 2);
residuals   = yE - hT * lambdaHat;
qHat        = residuals' * solveRD(residuals);
aT          = (nE - pD - pRho) / 2;
math.sigma2D = qHat / (2 * aT);

end