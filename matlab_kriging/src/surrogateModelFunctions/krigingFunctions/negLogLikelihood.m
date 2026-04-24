function [loss, gradZ] = negLogLikelihood(z, x, y, d, F)
% negLogLikelihood  Compute Concentrated NLL and its analytical gradient.
%
% Evaluates the objective function for the Ordinary Kriging hyperparameter 
% optimization. It analytically profiles the variance (sigma2) and the mean (mu).
% The gradient calculation incorporates the chain rule to account for the 
% logarithmic scaling of the theta parameters.
%
% INPUTS
%   z : [1 x (2d+1)] Decision vector [log10(theta_1..d), p_1..d, lambda].
%   x : [n x d] Input locations.
%   y : [n x 1] Target values.
%   d : Scalar number of input dimensions.
%
% OUTPUTS
%   loss  : Scalar Negative Log-Likelihood value.
%   gradZ : [1 x (2d+1)] Exact analytical gradient vector.

    % Extract parameters from decision vector
    theta  = 10.^z(1:d); % Return to linear space
    p      = z(d+1:2*d);
    lambda = z(end);
    n      = size(x, 1);
    
    % Correlation Matrix and Cholesky Factorization
    D = zeros(n, n);
    for k = 1:d
        D = D + theta(k) * (abs(x(:, k) - x(:, k)') .^ p(k));
    end
    R = exp(-D) + eye(n) * lambda;
    
    [L, flag] = chol(R, 'lower');
    if flag > 0
        loss = 1e6; gradZ = zeros(size(z)); return; % Matrix collapse protection
    end
    
    % Generalized Least Squares (GLS) and Concentrated Variance
    invPsi  = @(b) (L' \ (L \ b));

    % Resolve the system (F' * R^-1 * F) * muCoeffs = (F' * R^-1 * y)
    term1 = F' * invPsi(F);
    term2 = F' * invPsi(y);

    mu = term1 \ term2; 

    err = y - (F * mu);

    beta    = invPsi(err);
    sigma2  = (err' * beta) / n;
    
    % Scaled Objective Loss
    loss = 0.5 * (n * log(sigma2) + 2 * sum(log(diag(L))));
    
    % Exact Analytical Gradient
    if nargout > 1
        gradLogTheta = zeros(1, d);
        gradP        = zeros(1, d);
        
        W = invPsi(eye(n)) - (beta * beta') / sigma2;
        WR = W .* exp(-D);
        
        for k = 1:d
            deltaK = abs(x(:, k) - x(:, k)');
            
            % Gradient w.r.t log10(theta) via Chain Rule
            termTheta = deltaK .^ p(k);
            gradTheta = -sum(sum(WR .* termTheta));
            gradLogTheta(k) = gradTheta * theta(k) * log(10);
            
            % Gradient w.r.t p (safeguarded against log(0))
            mask = deltaK > 0; termP = zeros(n, n);
            termP(mask) = theta(k) * (deltaK(mask) .^ p(k)) .* log(deltaK(mask));
            gradP(k) = -sum(sum(WR .* termP));
        end
        gradLambda = trace(W);
        
        % Assemble scaled gradient vector
        gradZ = 0.5 * [gradLogTheta, gradP, gradLambda];
    end
end