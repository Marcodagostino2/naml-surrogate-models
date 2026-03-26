function F = buildPolynomialMatrix(x, degree)
% buildPolynomialMatrix  Generates the Universal Kriging design matrix F
%
% INPUTS
%   x      : [n x d] Input locations.
%   degree : 0 (Constant), 1 (Linear), 2 (Quadratic + Cross-terms)
%
% OUTPUT
%   F      : [n x q] Design matrix where q is the number of polynomial terms.

[n, d] = size(x);

if degree == 0
    F = ones(n, 1);

elseif degree == 1
    F = [ones(n, 1), x];

elseif degree == 2
    [I, J] = find(triu(ones(d, d)));

    F = [ones(n, 1), x, x(:, I) .* x(:, J)];

else
    error('Degree above 2 are not supported yet');
end
end