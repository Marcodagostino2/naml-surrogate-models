function coeffModel = mainCoefficientModel(options)
% mainCoefficientModel  Builds and evaluates aerodynamic surrogate models.
%
% INPUTS (name-value)
%   mode         : "multiFidelity" | "cfd" | "datcom"
%   splitPerc    : Train/test split ratio          (default 0.8)
%   reBuild      : Force dataset rebuild           (default true)
%   reTrain      : Force surrogate retraining      (default true)
%   saveDataset  : Persist dataset to disk         (default true)
%   saveModel    : Persist surrogate to disk       (default true)
%   plots        : Generate diagnostic plots       (default true)
%   exportPath   : Export folder for figures       (default auto timestamped)
%   exportFormat : "png" | "pdf"                   (default "png")
%   plotSurface  : Include surface/contour plots   (default true)
%
% OUTPUTS
%   coeffModel   : Trained CoefficientModel instance.
arguments
    options.mode         (1,1) string  = "cfd"
    options.splitPerc    (1,1) double  = 0.7
    options.reBuild      (1,1) logical = true
    options.reTrain      (1,1) logical = true
    options.saveDataset  (1,1) logical = false
    options.saveModel    (1,1) logical = false
    options.plots        (1,1) logical = true
    options.exportPath   (1,1) string  = ""
    options.exportFormat (1,1) string  = "png"
    options.plotSurface  (1,1) logical = true
end

%% PATHS
setupPaths();

%% INITIALIZATION & TRAINING
fprintf("\n=== INITIALIZING COEFFICIENT MODEL (%s) ===\n", upper(options.mode));
coeffModel = CoefficientModel( ...
    "mode",        options.mode,        ...
    "splitPerc",   options.splitPerc,   ...
    "reBuild",     options.reBuild,     ...
    "reTrain",     options.reTrain,     ...
    "saveDataset", options.saveDataset, ...
    "saveModel",   options.saveModel);

if ~coeffModel.isValid
    error("mainCoefficientModel:InvalidModel", ...
        "CoefficientModel initialization failed. Check constraints and datasets.");
end

%% DIAGNOSTICS & PLOTS
if options.plots
    fprintf("\n=== GENERATING DIAGNOSTIC PLOTS ===\n");

    exportPath = options.exportPath;
    if strlength(exportPath) == 0
        exportPath = fullfile("results", options.mode, ...
            string(datetime("now", "Format", "yyyyMMdd_HHmm")));
    end

    coeffModel.diagnose( ...
        "exportPath",   exportPath,          ...
        "exportFormat", options.exportFormat, ...
        "plotSurface",  options.plotSurface);

    fprintf("  Figures saved → %s\n", exportPath);
end

fprintf("\n=== SURROGATE WORKFLOW COMPLETE ===\n");
end

%% =========================================================================
function setupPaths()
    currentPath  = fileparts(mfilename('fullpath'));
    dissilePath  = fullfile(currentPath, '..', '..', 'dissileMatcom');
    commonPath   = fullfile(currentPath, '..', 'common');
    samplingPath = fullfile(commonPath, 'functions', 'surrogateModelFunctions', ...
        'samplingPlan', 'data');
    addpath(genpath(commonPath));
    addpath(genpath(dissilePath));
    addpath(genpath(samplingPath));
    addpath(genpath(currentPath));
end
