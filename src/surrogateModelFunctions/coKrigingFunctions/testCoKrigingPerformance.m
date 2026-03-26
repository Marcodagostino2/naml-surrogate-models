function testCoKrigingPerformance(predictor, model, nTest)
%TESTCOKRIGINGPERFORMANCE  Compute timing and memory for a test prediction
%
%   predictor : scalar function_handle @(x)
%   X         : training matrices (N x d)
%   mu        : scalar
%   corrVector, theta, p : Kriging parameters
%   nTest     : predictions to be performed

arguments
    predictor  function_handle
    model      struct
    nTest      double = 1e4
end

[rho, mu, corrVector, xC, thetaC, pC, sigma2C, lambdaC, ...
    xE, thetaD, pD, sigma2D, lambdaD] = unpackCoKrigingModel(model);

[NLF, d] = size(xC);
NHF = size(xE, 1);

% Model memory
vars = whos('rho','mu','corrVector','xC', 'thetaC', 'pC', 'sigma2C', 'lambdaC', ...
    'xE', 'thetaD', 'pD', 'sigma2D', 'lambdaD');
bytesModel = sum([vars.bytes]);
MBModel    = bytesModel / 1024^2;

fprintf('=== CO-KRIGING MODEL MEMORY FOOTPRINT ===\n');
fprintf('Low-fidelity Training samples N      : %d\n', NLF);
fprintf('High-fidelity Training samples N      : %d\n', NHF);
fprintf('Input dimension d       : %d\n', d);
fprintf('Approx. model memory    : %.3f MB\n\n', MBModel);

% Test points generation
xMin = min(xE, [], 1);
xMax = max(xE, [], 1);
xTest = rand(nTest, d) .* (xMax - xMin) + xMin;

% Time measuring for n predictions
tic;
predictor(xTest);
elapsed = toc;

timePerPred = elapsed / nTest;
fprintf('=== KRIGING PREDICTION TIMING ===\n');
fprintf('Number of test queries  : %d\n', nTest);
fprintf('Total elapsed time      : %.4f s\n', elapsed);
fprintf('Avg time per prediction : %.6e s\n\n', timePerPred);

end