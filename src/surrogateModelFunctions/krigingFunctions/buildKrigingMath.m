function math = buildKrigingMath(x, y, theta, p, lambda, polyDegree, dy, dyDirs)
% buildKrigingMath  Construct the matrices required for Kriging prediction.
%
% Supports both standard Kriging and Gradient-Enhanced Kriging (GEK).
% If gradient data (dy and dyDirs) is provided, the function automatically
% augments the mathematical structures to build a GEK model.
%
% INPUTS
%   x          : [n x d] Matrix of unit-space input coordinates.
%   y          : [n x 1] Vector of target output values.
%   theta      : [1 x d] Correlation length parameters.
%   p          : [1 x d] Exponent parameters.
%   lambda     : [1 x 1] Regularization/nugget parameter.
%   polyDegree : [1 x 1] Polynomial trend degree (default = 0).
%   dy         : [n x m] (Optional) Matrix of target gradients.
%   dyDirs     : [1 x m] (Optional) Active gradient dimensions.
%
% OUTPUT
%   math : Struct containing derived L, mu, beta, etc.

arguments
    x           (:, :) double
    y           (:, 1) double
    theta       (1, :) double
    p           (1, :) double
    lambda      (1, 1) double
    polyDegree  (1, 1) double = 0
    dy          (:, :) double = []
    dyDirs      (1, :) double = []
end

n = size(x, 1);

if ~isempty(dy) && ~isempty(dyDirs)
    % GEK Mode
    p(dyDirs) = 2; % Force mathematical consistency for differentiability
    
    yAug = [y; dy(:)];
    fBase = buildPolynomialMatrix(x, polyDegree);
    fAug  = buildPolynomialMatrixGEK(x, polyDegree, dyDirs, fBase);
    
    [~, L, ~] = buildCovarianceMatrixGEK(x, theta, p, lambda, dyDirs);
    
    invPsi = @(b) (L' \ (L \ b));  
    term1  = fAug' * invPsi(fAug);
    term2  = fAug' * invPsi(yAug);
    mu     = term1 \ term2;
    err    = yAug - (fAug * mu);
    beta   = invPsi(err);
    sigma2 = (err' * beta) / length(yAug);
    
    math.isGEK  = true;
    math.dyDirs = dyDirs;
    
else
    % Standard OK/UK Mode
    fBase = buildPolynomialMatrix(x, polyDegree);
    L = buildStableCorrelationMatrix(x, theta, p, lambda);
    
    invPsi = @(b) (L' \ (L \ b));
    term1  = fBase' * invPsi(fBase);
    term2  = fBase' * invPsi(y);
    mu     = term1 \ term2;
    err    = y - (fBase * mu);
    beta   = invPsi(err);
    sigma2 = (err' * beta) / n;
    
    math.isGEK = false;
    math.dyDirs = [];
end

math.L          = L;
math.mu         = mu;
math.sigma2     = sigma2;
math.beta       = beta;
math.polyDegree = polyDegree;

end