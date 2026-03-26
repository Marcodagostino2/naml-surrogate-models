function [J, distinctDistance] = jd(X, metric)
% Computes the distances between all pairs of points in a sampling plan X
% using the p-norm, sorts  them  in  ascending order and removes multiple 
% occurences. 
%
% Inputs:
%       X - sampling plan being evaluated
%       p - distance norm (p = 1 rectangular - default, p = 2 Euclidean)
%
% Outputs:
%       J - multiplicity array (that is, the number of pairs separated by 
%           each distance value).
%       distinctDistance - list of distinct distance values

d = pdist(X, metric);

% Remove multiple occurences
[distinctDistance, ~, ic] = unique(d);

% Generate multiplicity array using accumarray
J = accumarray(ic, 1)';

