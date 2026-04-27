% COEFFICIENT HYPERPARAMETER OPTIMIZATION SETTINGS
% Optimization via Multi-Start Gradient-Enhanced SQP with GLS Profiling
% Decision vector: z = [log10(theta_1..theta_d), p_1..p_d, lambda]

optimizationOptions.nInputs  = 5;
optimizationOptions.nOutputs = 1;

% --- SQP OPTIMIZER & MULTI-START SETTINGS ---
optimizationOptions.numStarts        = 1;        % Number of random initialization points
optimizationOptions.fminconAlgorithm = 'sqp';    % Optimization algorithm ('sqp' or 'interior-point')
optimizationOptions.fminconOptTol    = 1e-6;     % First-order optimality tolerance
optimizationOptions.fminconStepTol   = 1e-6;     % Step size tolerance
optimizationOptions.fminconMaxIter   = 1000;     % Maximum iterations per start
optimizationOptions.fminconDisplay   = 'off';    % Console display level

% KRIGING HYPERPARAMETER OPTIMIZATION SETTINGS

% --- POLYNOMIAL TREND OPTIONS ---
optimizationOptions.polyDegree = 0;

% Correlation length bounds (optimized in log10 space)
optimizationOptions.lbLogTheta = -5 * ones(1, optimizationOptions.nInputs);
optimizationOptions.ubLogTheta =  3 * ones(1, optimizationOptions.nInputs);

% Correlation kernel exponent bounds
optimizationOptions.lbP        = 1.5 * ones(1, optimizationOptions.nInputs);
optimizationOptions.ubP        = 2 * ones(1, optimizationOptions.nInputs);

% Nugget / regularization parameter bounds
optimizationOptions.lbLambda   = 1e-7;
optimizationOptions.ubLambda   = 1e-2;

% CO-KRIGING HYPERPARAMETER OPTIMIZATION SETTINGS
% Note: The scaling factor (rho) is profiled analytically inside the solver

% --- POLYNOMIAL TREND OPTIONS ---
optimizationOptions.polyDegreeC   = 2;
optimizationOptions.polyDegreeD   = 0;
optimizationOptions.polyDegreeRho = 0;

% --- COARSE MODEL (LOW-FIDELITY) BOUNDS ---
% Correlation length bounds (optimized in log10 space)
optimizationOptions.lbLogThetaC = -5 * ones(1, optimizationOptions.nInputs);
optimizationOptions.ubLogThetaC =  3 * ones(1, optimizationOptions.nInputs);

% Correlation kernel exponent bounds
optimizationOptions.lbPC        = 1.5 * ones(1, optimizationOptions.nInputs);
optimizationOptions.ubPC        = 2 * ones(1, optimizationOptions.nInputs);

% Nugget / regularization parameter bounds
optimizationOptions.lbLambdaC   = 1e-10;
optimizationOptions.ubLambdaC   = 1e-5;

% --- DISCREPANCY MODEL (HF - rho*LF) BOUNDS ---
% Correlation length bounds (optimized in log10 space)
optimizationOptions.lbLogThetaD = -5 * ones(1, optimizationOptions.nInputs);
optimizationOptions.ubLogThetaD =  3 * ones(1, optimizationOptions.nInputs);

% Correlation kernel exponent bounds
optimizationOptions.lbPD        = 1.5 * ones(1, optimizationOptions.nInputs);
optimizationOptions.ubPD        = 2 * ones(1, optimizationOptions.nInputs);

% Nugget / regularization parameter bounds
optimizationOptions.lbLambdaD   = 1e-6;
optimizationOptions.ubLambdaD   = 1e-2;