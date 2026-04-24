function [theta, p, lambda] = unpackKrigingParameters(param, d)
% unpackKrigingParameters  Decode the GA decision vector into Kriging hyperparameters.
%
% The GA optimizes a single parameter vector with the following layout:
%   param = [log10(theta_1..theta_d), p_1..p_d, lambda]
%
% This function converts that vector into:
%   - theta  : [1 x d] correlation length scales (theta_i > 0), reconstructed as 10.^log10(theta_i)
%   - p      : [1 x d] correlation exponents (typically in [1,2] for a power-exponential kernel)
%   - lambda : scalar nugget/regularization term
%
% INPUTS
%   param : numeric vector of length (2*d + 1)
%   d     : number of input dimensions
%
% OUTPUTS
%   theta  : [1 x d] vector
%   p      : [1 x d] vector
%   lambda : scalar
theta  = 10.^param(1:d);
p      = param(d+1:end-1);
lambda = param(end);
end

