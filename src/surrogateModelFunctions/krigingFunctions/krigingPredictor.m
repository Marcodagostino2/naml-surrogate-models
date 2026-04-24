function [yPred, varPred] = krigingPredictor(xQuery, data, hyperparams, math)
% krigingPredictor  Evaluate the Kriging surrogate model at query points.
%
% Seamlessly supports both standard Kriging and Gradient-Enhanced Kriging 
% based on the flags provided in the math struct. Computes the predictive 
% mean and, optionally, the Universal Kriging predictive variance.
%
% INPUTS
%   xQuery      : [m x d] Matrix of query points.
%   data        : Struct containing training data (.x required).
%   hyperparams : Struct containing optimized parameters.
%   math        : Struct containing derived matrices.
%
% OUTPUTS
%   yPred       : [m x 1] Predicted responses (mean).
%   varPred     : [m x 1] Predicted variance (Universal Kriging formulation).

arguments
    xQuery      (:, :) double
    data        (1, 1) struct
    hyperparams (1, 1) struct
    math        (1, 1) struct
end

xTrain = data.x;

theta = hyperparams.theta;
p     = hyperparams.p;
mu    = math.mu;
beta  = math.beta;

fQuery = buildPolynomialMatrix(xQuery, math.polyDegree);

if isfield(math, 'isGEK') && math.isGEK
    % Force consistency for active directions
    p(math.dyDirs) = 2; 
    
    % Compute augmented cross-correlation vector (values + gradients)
    psiAug = crossCorrMatrixGEK(xTrain, xQuery, theta, p, math.dyDirs);
    yPred  = fQuery * mu + psiAug' * beta;
    
    % ---------------------------------------------------------------------
    % GEK PREDICTIVE VARIANCE (Simple Kriging approximation)
    % ---------------------------------------------------------------------
    if nargout > 1
        lMat = math.l; % Cholesky factor of the augmented correlation matrix
        invRPsiAug = lMat' \ (lMat \ psiAug);
        varGP = 1 - sum(psiAug .* invRPsiAug, 1)';
        varGP(varGP < 0) = 0; % Numerical safety
        
        varPred = math.sigma2 * varGP;
    end
    
else
    % Standard cross-correlation vector
    psi   = crossCorrMatrix(xTrain, xQuery, theta, p);
    yPred = fQuery * mu + psi' * beta;
    
    % ---------------------------------------------------------------------
    % STANDARD PREDICTIVE VARIANCE (Universal Kriging formulation)
    % ---------------------------------------------------------------------
    if nargout > 1
        % Extract Cholesky factorization
        if isfield(math, 'l')
            lMat = math.l; 
        elseif isfield(math, 'L')
            lMat = math.L; 
        else
            error('krigingPredictor:MissingCholesky', ...
                  'Cholesky factor missing in math struct. Cannot compute variance.');
        end
        
        % Base Kriging spatial variance: 1 - r(x)^T * R^-1 * r(x)
        invRPsi = lMat' \ (lMat \ psi);
        varGP   = 1 - sum(psi .* invRPsi, 1)';
        varGP(varGP < 0) = 0; % Numerical safety
        
        % Universal Kriging inflation due to trend uncertainty
        fTrain  = buildPolynomialMatrix(xTrain, math.polyDegree);
        invRF   = lMat' \ (lMat \ fTrain);
        
        % Sigma_beta = (F^T * R^-1 * F)^-1
        covBeta = (fTrain' * invRF) \ eye(size(fTrain, 2));
        
        % u(x) = f(x) - F^T * R^-1 * r(x)
        uQuery = fQuery - (invRF' * psi)';
        varTrend = sum((uQuery * covBeta) .* uQuery, 2);
        
        % Total predictive variance
        varPred = math.sigma2 * (varGP + varTrend);
    end
end

end