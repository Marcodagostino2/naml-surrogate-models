function [loss, gradZ, betaRho, muD] = coNegLogLikelihood(z, xE, yE, yCAtE, d, fD, fRho)
% coNegLogLikelihood  Compute Discrepancy NLL with GLS Profiling (Le Gratiet formulation).
%
% Evaluates the objective function for the Co-Kriging discrepancy model.
% The spatially variable scaling factor rho(x) = fRho * betaRho and the 
% discrepancy mean muD are optimally profiled out analytically using GLS.
%
% INPUTS
%   z       : [1 x (2d+1)] Decision vector [log10(thetaD), pD, lambdaD].
%   xE      : [nE x d] Expensive input locations.
%   yE      : [nE x 1] Expensive target values.
%   yCAtE   : [nE x 1] Cheap model predictions at expensive locations.
%   d       : Scalar number of input dimensions.
%   fD      : [nE x pD] Polynomial design matrix for discrepancy trend.
%   fRho    : [nE x pRho] Polynomial design matrix for the scaling factor.
%
% OUTPUTS
%   loss    : Scalar Negative Log-Likelihood value.
%   gradZ   : [1 x (2d+1)] Exact analytical gradient vector.
%   betaRho : [pRho x 1] Analytically computed scaling coefficients.
%   muD     : [pD x 1] Analytically computed discrepancy mean coefficients.

    % Extract discrepancy parameters (betaRho is profiled, NOT in z)
    theta  = 10.^z(1:d);
    p      = z(d+1:2*d);
    lambda = z(end);
    n      = size(xE, 1);
    
    % Correlation Matrix and Cholesky Factorization
    D = zeros(n, n);
    for k = 1:d
        D = D + theta(k) * (abs(xE(:, k) - xE(:, k)') .^ p(k));
    end
    R = exp(-D) + eye(n) * lambda;
    
    [L, flag] = chol(R, 'lower');
    if flag > 0
        loss = 1e6; gradZ = zeros(size(z)); betaRho = 1; muD = 0; return;
    end
    invPsi = @(b) (L' \ (L \ b));
    
    % UNIVERSAL KRIGING PROFILING: Analytical GLS for [muD, betaRho]
    % rho(x) * yC(x) = (fRho * betaRho) .* yC = (fRho .* yC) * betaRho
    H = [fD, fRho .* yCAtE]; 
    term1 = H' * invPsi(H);
    term2 = H' * invPsi(yE);
    B = term1 \ term2; 
    
    % Extract generalized estimators
    pD = size(fD, 2);
    muD = B(1:pD);
    betaRho = B(pD+1:end);     
    
    % Centered discrepancy error
    errD    = yE - (H * B);
    beta    = invPsi(errD);
    sigma2  = (errD' * beta) / n;
    
    % Scaled Objective Loss
    loss = 0.5 * (n * log(sigma2) + 2 * sum(log(diag(L))));
    
    % Exact Analytical Gradient (Envelope theorem ensures gradients are unaltered by GLS)
    if nargout > 1
        gradLogTheta = zeros(1, d); gradP = zeros(1, d);
        W  = invPsi(eye(n)) - (beta * beta') / sigma2;
        WR = W .* exp(-D);
        
        for k = 1:d
            deltaK = abs(xE(:, k) - xE(:, k)');
            
            % Chain rule for log10(theta)
            termTheta = deltaK .^ p(k);
            gradTheta = -sum(sum(WR .* termTheta));
            gradLogTheta(k) = gradTheta * theta(k) * log(10); 
            
            % Derivative w.r.t p
            mask = deltaK > 0; termP = zeros(n, n);
            termP(mask) = theta(k) * (deltaK(mask) .^ p(k)) .* log(deltaK(mask));
            gradP(k) = -sum(sum(WR .* termP));
        end
        gradLambda = trace(W);
        
        % Assemble scaled gradient vector
        gradZ = 0.5 * [gradLogTheta, gradP, gradLambda];
    end
end