function coeffModel = mainCoefficientModel(options)
% mainCoefficientModel  Builds and evaluates aerodynamic surrogate models.
%
% This function serves as the primary entry point to initialize, train, and 
% evaluate multi-fidelity (Co-Kriging) or single-fidelity (Kriging) surrogate 
% models for aerodynamic coefficients. It orchestrates the workflow by 
% delegating dataset generation, training, and gridding to the CoefficientModel class.
%
% INPUTS
%   options     : Name-value pairs for configuration:
%                 - mode        (string)  : Dataset mode ("multiFidelity", "cfd", "datcom").
%                 - splitPerc   (double)  : Train/test split ratio (default 0.8).
%                 - reBuild     (logical) : Force dataset rebuild.
%                 - reTrain     (logical) : Force surrogate model retraining.
%                 - saveModel   (logical) : Save the trained surrogate to disk.
%                 - plots       (logical) : Generate diagnostic plots.
%
% OUTPUTS
%   coeffModel  : Initialized CoefficientModel object containing the trained 
%                 surrogate.

arguments
    options.mode         (1,1)   string  = "multiFidelity" 
    options.splitPerc    (1,1)   double  = 0.85
    options.reBuild      (1,1)   logical = true
    options.reTrain      (1,1)   logical = true
    options.saveDataset  (1,1)   logical = true
    options.saveModel    (1,1)   logical = true
    options.plots        (1,1)   logical = true
end

%% PATHS & MISSION SETUP
setupPaths();

%% COEFFICIENT MODEL INITIALIZATION & TRAINING
fprintf("\n=== INITIALIZING COEFFICIENT MODEL (%s) ===\n", upper(options.mode));
coeffModel = CoefficientModel( ...
    "mode", options.mode, ...
    "splitPerc", options.splitPerc, ...
    "reBuild", options.reBuild, ...
    "reTrain", options.reTrain, ...
    "saveDataset", options.saveDataset, ...
    "saveModel", options.saveModel);

if ~coeffModel.isValid
    error('CoefficientModel initialization failed. Check constraints and datasets.');
end

%% BENCHMARKING (AERODECK LOOKUP)
fprintf("\n=== EVALUATION & BENCHMARKING ===\n");
nQuery = 1e4;
nInputs = length(coeffModel.inNames);
xTestUnit = rand(nQuery, nInputs); 

tic;
yPred = coeffModel.model.predict(xTestUnit); 
tElapsed = toc;

fprintf("Benchmarking evaluation speed (1e4 predictions via AeroDeck)...\n");
fprintf("  Total time for %d multi-output evaluations: %.4f seconds\n", nQuery, tElapsed);
fprintf("  Average time per prediction per output: %.2e seconds\n", tElapsed / (nQuery * size(yPred, 2)));

%% DIAGNOSTICS & PLOTS
if options.plots
    fprintf("\n=== GENERATING DIAGNOSTIC PLOTS ===\n");
    coeffModel.plotDiagnostics();
end

fprintf("\n=== MULTI-OUTPUT SURROGATE WORKFLOW COMPLETE ===\n");
end

function setupPaths()
    % setupPaths Configures MATLAB path dependencies for the surrogate framework.
    currentPath  = fileparts(mfilename('fullpath'));
    dissilePath  = fullfile(currentPath, '..', '..', 'dissileMatcom');
    commonPath   = fullfile(currentPath, '..', 'common');
    samplingPath = fullfile(commonPath, 'functions', 'surrogateModelFunctions', 'samplingPlan', 'data');
    addpath(genpath(commonPath));
    addpath(genpath(dissilePath));
    addpath(genpath(samplingPath));
    addpath(genpath(currentPath));
end
