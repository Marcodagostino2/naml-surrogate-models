function phiQ = maxminPhi(X, metric, q)
% Calculates the sampling plan quality criterion of Morris and Mitchell.
%
% Inputs:
%       X - sampling plan
%       q - exponent used in the calculation of the metric
%       metric - the distance metric to be used (p=1 rectangular - default, p=2
%           Euclidean)
%
% Output:
%       phiQ - sampling plan `space-fillingness' metric

% Assume defaults if arguments list incomplete
if ~exist('q','var')
    q = 2;
end

% Calculate the distances between all pairs of
% points (using the p-norm) and build multipli-
% city array J
[J,d] = jd(X,metric);

% The sampling plan quality criterion
phiQ = sum(J.*(d.^(-q)))^(1/q);