function loss = negLogLikelihoodGEK(z, x, y, dy, dyDirs, d, fBase, polyDegree)
% negLogLikelihoodGEK  Compute Concentrated NLL for Gradient-Enhanced Kriging.
%
% INPUTS
%   z          : [1 x (2d+1)] Decision vector [log10(theta), p, lambda].
%   x          : [n x d] Input locations.
%   y          : [n x 1] Target values.
%   dy         : [n x m] Target gradients.
%   dyDirs     : [1 x m] Active gradient dimensions.
%   d          : Scalar number of input dimensions.
%   fBase      : [n x nPoly] Standard design matrix.
%   polyDegree : Degree of polynomial trend.
%
% OUTPUT
%   loss       : Scalar Negative Log-Likelihood value.

    % Extract parameters
    theta  = 10.^z(1:d); 
    p      = z(d+1:2*d);
    lambda = z(end);
    
    % FORCE p=2 for active gradient directions to guarantee 
    % mathematical twice-differentiability at dx=0
    p(dyDirs) = 2; 

    % Build Augmented Covariance Matrix
    [~, LAug, flag] = buildCovarianceMatrixGEK(x, theta, p, lambda, dyDirs);
    if flag > 0
        loss = 1e6; return; % Matrix collapse protection
    end
    
    % Stack y and dy column by column -> [y; dy_1; dy_2; ...]
    yAug = [y; dy(:)]; 
    
    % Stack F and dF/dx
    fAug = buildPolynomialMatrixGEK(x, polyDegree, dyDirs, fBase);
    
    invPsi = @(b) (LAug' \ (LAug \ b));
    term1  = fAug' * invPsi(fAug);
    term2  = fAug' * invPsi(yAug);
    mu     = term1 \ term2; 
    
    err    = yAug - (fAug * mu);
    beta   = invPsi(err);
    sigma2 = (err' * beta) / length(yAug);
    
    loss = 0.5 * (length(yAug) * log(sigma2) + 2 * sum(log(diag(LAug))));
end


