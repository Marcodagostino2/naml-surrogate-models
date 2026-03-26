function yPred = krigingPredictor(xQuery, data, hyperparams, math)
% krigingPredictor  Evaluate the Kriging surrogate model at query points.
%
% Seamlessly supports both standard Kriging and Gradient-Enhanced Kriging 
% based on the flags provided in the math struct.
%
% INPUTS
%   xQuery      : [m x d] Matrix of query points.
%   data        : Struct containing training data.
%   hyperparams : Struct containing optimized parameters.
%   math        : Struct containing derived matrices.
%
% OUTPUT
%   yPred       : [m x 1] Predicted responses.

arguments
    xQuery      (:, :) double
    data        (1, 1) struct
    hyperparams (1, 1) struct
    math        (1, 1) struct
end

xTrain = data.x;
theta  = hyperparams.theta;
p      = hyperparams.p;
mu     = math.mu;
beta   = math.beta;

fQuery = buildPolynomialMatrix(xQuery, math.polyDegree);

if isfield(math, 'isGEK') && math.isGEK
    % Force consistency for active directions
    p(math.dyDirs) = 2; 
    
    % Compute augmented cross-correlation vector (values + gradients)
    psiAug = crossCorrMatrixGEK(xTrain, xQuery, theta, p, math.dyDirs);
    yPred = fQuery * mu + psiAug' * beta;
else
    % Standard cross-correlation vector
    psi = crossCorrMatrix(xTrain, xQuery, theta, p);
    yPred = fQuery * mu + psi' * beta;
end

end



