function Psi = corrMatrix(x, theta, p)
% CORRMATRIX  Builds the correlation matrix (psi) for Kriging interpolation.
%
%   psi = corrMatrix(x, theta, p)
%
%   Inputs:
%       x       : matrix (n x d), sample points
%       theta   : 1 x d vector of correlation parameters
%       p       : 1 x d vector of exponents for the correlation function
%
%   Output:
%       Psi     : n x n correlation matrix

%% Initialize correlation matrix
n = size(x, 1);                     % Number of data points
Psi = zeros(n, n, 'like', x);       % Preallocate

%% Construct the correlation matrix (vectorized over dimensions)
for k = 1:length(theta)
    diff = abs(x(:, k) - x(:, k)');   % Absolute differences along the k-th dimension (n x n)
    Psi = Psi + theta(k) * diff.^p(k);% Accumulate weighted sum for all dimensions
end

%% Apply exponential correlation function
Psi = exp(-Psi);    % Compute final correlation values

%% Set diagonal elements to 1 (perfect self-correlation)
Psi(1:n+1:end) = 1; % MATLAB trick for accessing diagonal indices

end

