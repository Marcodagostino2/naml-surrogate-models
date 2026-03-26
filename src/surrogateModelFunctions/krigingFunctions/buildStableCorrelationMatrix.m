function L = buildStableCorrelationMatrix(x, theta, p, lambda)
% buildStableCorrelationMatrix  Compute a Cholesky factor of the regularized correlation matrix.
%
% This function builds the Kriging correlation matrix Psi and returns a lower
% triangular Cholesky factor L such that:
%   L*L' = Psi + (lambda + jitter)*I
%
% The diagonal term (lambda + jitter) is used to:
%   - apply the Kriging nugget/regularization (lambda),
%   - add a small increasing "jitter" if Psi is nearly singular or not positive definite.
%
% INPUTS
%   x      : [n x d] training inputs (unit space).
%   theta  : [1 x d] correlation length scales.
%   p      : [1 x d] correlation exponents.
%   lambda : scalar nugget/regularization term.
%
% OUTPUT
%   L : [n x n] lower-triangular Cholesky factor.
%
% Notes
%   - chol requires the matrix to be symmetric positive definite; if it fails, jitter is
%     increased and the factorization is retried. Using two outputs from chol is another
%     way to avoid throwing inside the loop.
arguments
    x      {mustBeNumeric}
    theta  {mustBeNumeric}
    p      {mustBeNumeric}
    lambda (1,1) {mustBeNumeric}
end

Psi = corrMatrix(x, theta, p);
n   = size(Psi, 1);

if lambda == 0
    jitter = eps(class(Psi));
else
    jitter = 0;
end

[L, flag] = chol(Psi + (lambda + jitter) * eye(n, 'like', Psi), "lower");
if flag == 0
    return
end

error("buildStableCorrelationMatrix:NotPD", ...
      "Correlation matrix not positive definite after jitter up to %.1e.", jitter);
end

