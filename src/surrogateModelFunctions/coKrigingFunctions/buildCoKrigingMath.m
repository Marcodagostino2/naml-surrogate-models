function math = buildCoKrigingMath(xC, yC, xE, yE, hypC, mathC, hypD, mathD, rho)
% buildCoKrigingMath  Construct the augmented matrices required for Co-Kriging prediction.
%
% This function builds the augmented covariance matrix C for the stacked 
% observations yAug = [yC; yE]. It computes the Cholesky factorization, 
% estimates the global mean (mu) via Generalized Least Squares (GLS), and 
% computes the pre-weighted beta vector (corrVector) used for fast predictions.
%
% INPUTS
%   xC    : [nC x d] Cheap training inputs.
%   yC    : [nC x 1] Cheap training outputs.
%   xE    : [nE x d] Expensive training inputs.
%   yE    : [nE x 1] Expensive training outputs.
%   hypC  : Struct containing tuned hyperparameters for the cheap GP.
%   mathC : Struct containing derived math parameters (including sigma2) for cheap GP.
%   hypD  : Struct containing tuned hyperparameters for the discrepancy GP.
%   mathD : Struct containing derived math parameters (including sigma2) for discrepancy GP.
%   rho   : Scalar autoregressive scaling factor.
%
% OUTPUT
%   math : Struct containing:
%                - L       : Cholesky factorization of the augmented covariance.
%                - mu      : Estimated global mean.
%                - sigma2  : Variance scalar of the model
%                - alpha   : C^-1 * 1 Precomputed inverse vector product for fast prediction
%                - beta    : Precomputed inverse vector product (formerly corrVector).
arguments
    xC          (:, :) double
    yC          (:, 1) double
    xE          (:, :) double
    yE          (:, 1) double
    hypC        (1, 1) struct
    mathC       (1, 1) struct
    hypD        (1, 1) struct
    mathD       (1, 1) struct
    rho         (1, 1) double
end

nC = size(xC, 1);
nE = size(xE, 1);
nTot = nC + nE;

% Extract variance scalings from the underlying Kriging math structs
sigma2C = mathC.sigma2;
sigma2D = mathD.sigma2;

% Build basic correlation matrices using parameters from hypC and hypD
PsiCC = corrMatrix(xC, hypC.theta, hypC.p);
PsiEE = corrMatrix(xE, hypC.theta, hypC.p);
PsiDD = corrMatrix(xE, hypD.theta, hypD.p);

% Efficient cross-correlation between cheap and expensive locations
PsiCE = crossCorrMatrix(xC, xE, hypC.theta, hypC.p);

% Assemble augmented covariance matrix C with nugget (lambda) regularization
cov11 = sigma2C * (PsiCC + hypC.lambda * eye(nC, class(xC)));
cov12 = rho * sigma2C * PsiCE;
cov21 = cov12.';
cov22 = (rho^2) * sigma2C * PsiEE + sigma2D * (PsiDD + hypD.lambda * eye(nE, class(xE)));

C = [cov11, cov12; 
     cov21, cov22];

% Stable Cholesky Factorization
[L, flag] = chol(C, "lower"); % Try strictly first

% Add jitter if standard factorization fails and no nugget was provided
if flag ~= 0 && (hypC.lambda == 0 && hypD.lambda == 0)
    jitter = eps(class(C));
    [L, flag] = chol(C + jitter * eye(nTot, class(C)), "lower");
end

if flag ~= 0
    error("buildCoKrigingMath:NotPD", ...
        "Augmented covariance matrix is not positive definite. Check lambda bounds or duplicate samples.");
end

solveCinv = @(b) (L' \ (L \ b));  

yAug = [yC; yE];
muD  = mathD.mu; 

% Polynmial design matrices
fCC = buildPolynomialMatrix(xC, mathC.polyDegree);
fCE = buildPolynomialMatrix(xE, mathC.polyDegree);
fDE = buildPolynomialMatrix(xE, mathD.polyDegree);

% Matrici aumentate H e K per il calcolo GLS
H = [fCC; rho * fCE]; 
K = [zeros(nC, size(fDE, 2)); fDE];

% Aggiornamento Bayesiano del trend LF (muC) con i dati HF
yAdjusted = yAug - K * muD;
alpha = solveCinv(H);
term1 = H' * alpha;
term2 = H' * solveCinv(yAdjusted);
muC   = term1 \ term2;

yCentered = yAug - (H * muC + K * muD);
beta      = solveCinv(yCentered);

% Assemble final math structure
math.L           = L;
math.muC         = muC; 
math.muD         = muD; 
math.polyDegreeC = mathC.polyDegree; 
math.polyDegreeD = mathD.polyDegree; 
math.alpha       = alpha;
math.beta        = beta;
math.sigma2C     = sigma2C;
math.sigma2D     = sigma2D;

end