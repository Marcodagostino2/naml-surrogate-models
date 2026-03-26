function x = perturb(x, pertNum)
% Interchanges pairs of randomly chosen elements within randomly chosen
% columns of a sampling plan a number of times. If the plan is a Latin
% hypercube, the result of this operation will also be a Latin hypercube.
%
% Inputs:
%       x - sampling plan
%       pertNum - the number of changes (perturbations) to be made to X.
%
% Output:
%       x - perturbed sampling plan

if ~exist('pertNum','var')
	pertNum = 1;
end

[n, k] = size(x);

% compute distance impact per column
colScore = zeros(1, k);
for j = 1:k
    colScore(j) = std(x(:,j)); % rough proxy of dispersion
end

% avoid zero-probability columns
prob = colScore / sum(colScore);
if any(prob < 0)
    prob = ones(1, k)/k;
end

for perturbCount = 1: pertNum
    
    % choose column with weighted probability
    col = randsample(k, 1, true, prob);

    % pick two distinct rows
    el1 = randi(n); 
    el2 = randi(n);
    while el1 == el2
        el2 = randi(n);
    end

    % swap
    tmp = x(el1, col);
    x(el1, col) = x(el2, col);
    x(el2, col) = tmp;
end