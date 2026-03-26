function [samplingPlan, valPhi] = mainSampling(settings, options)
% MAINSAMPLING  Generates and optimizes a Latin Hypercube Sampling (LHS) plan.
%
%   This function constructs an optimized sampling plan for surrogate-model
%   training, uncertainty quantification, or general design–space exploration.
%   The sampling is based on a Morris–Mitchell criterion applied to Latin
%   Hypercube designs, using multiple candidate sets and iterative refinement.
%
%   INPUT:
%       settings : (optional) Settings object or struct defining the sampling
%                  configuration. If empty, a default 'sampling' setup is loaded.
%       options  : Optional overrides for specific sampling parameters such as:
%                  • nPoints        – number of sampling points
%                  • nDim           – dimensionality of design space
%                  • nCandidates    – number of random LHS candidates
%                  • maxIterations  – refinement steps per candidate
%                  • metric         – distance metric for Φ_q criterion
%                  • q              – exponent for Φ_q computation
%                  • sortCriterion  – ranking rule for candidate plans
%                  • enablePlots    – flag for 2D visualization
%                  • savePlan       – flag for saving the final plan
%
%   OUTPUT:
%       samplingPlan : Optimized sampling matrix of size (nPoints × nDim)
%       valPhi       : Final Φ_q metric value of the optimal plan
%
%   DEPENDENCIES:
%       - Settings.m
%       - bestLatinHypercube.m
%
%   USAGE EXAMPLE:
%       [plan, phi] = mainSampling(Settings('sampling'));

arguments
    settings               Settings   = Settings.empty;
    options.nPoints        double     = [];
    options.nDim           double     = [];
    options.nCandidates    double     = [];
    options.maxIterations  double     = [];   
    options.metric         string     = [];
    options.q              double     = [];
    options.sortCriterion  string     = [];
    options.enablePlots    logical    = false;
    options.preLoadedData  logical    = [];
    options.savePlan       logical    = [];
    options.seed           double     = [];
end

%% PATH INITIALIZATION
% Determine function location and add necessary folders to MATLAB path.
currentPath = fileparts(mfilename("fullpath"));
dataPath    = fullfile(currentPath, 'data');

addpath(genpath(dataPath));
addpath(genpath(currentPath));

%% LOAD SETTINGS
% Create default settings if none are provided and apply user overrides.
if isempty(settings)
    settings = Settings('sampling'); 
end
Settings.read(settings, options, 'sampling');

%% LOAD OR GENERATE SAMPLING PLAN
dataFile = fullfile(dataPath, settings.sampling.fileName);

% Extract sampling settings
nDim          = settings.sampling.nDim;
nPoints       = settings.sampling.nPoints;
qPhiSort      = settings.sampling.q;
nCandidates   = settings.sampling.nCandidates;
maxIterations = settings.sampling.maxIterations;
sortCriterion = settings.sampling.sortCriterion;
metric        = settings.sampling.metric;
qOptimization = settings.sampling.qOptimization;

rng(settings.sampling.seed)

% --- CASE 1: try to load precomputed data ---
if isfield(settings.sampling,"preLoadedData") && settings.sampling.preLoadedData
    
    if isfile(dataFile)
        loaded = load(dataFile, "samplingPlan", "nDim", "nPoints");

        % Check dimensional consistency
        if loaded.nDim == nDim && loaded.nPoints == nPoints
            fprintf("[INFO] Loaded sampling plan from file.\n");
            samplingPlan = loaded.samplingPlan;

            % Compute Phi on the loaded plan
            valPhi = maxminPhi(samplingPlan, metric, qPhiSort);

            return; 
        else
            fprintf("[WARNING] Saved sampling plan dimensions do not match.\n");
            fprintf("          Saved: %d points, dim %d\n", loaded.nPoints, loaded.nDim);
            fprintf("          Required: %d points, dim %d\n", nPoints, nDim);
            fprintf("          Recomputing sampling plan...\n");
        end
    else
        fprintf("[WARNING] preLoadedData enabled but sampling file not found.\n");
        fprintf("          Computing sampling plan...\n");
    end
end

% --- CASE 2: compute plan from scratch ---
fprintf("[INFO] Computing optimal Latin Hypercube Design...\n");
tStart = tic;

[samplingPlan, qBest, valPhi] = bestLatinHypercube( ...
    nPoints, nDim, nCandidates, maxIterations, metric, ...
    qPhiSort, sortCriterion, qOptimization);

fprintf("[INFO] Done in %.2f seconds.\n", toc(tStart));

fprintf("Optimum plan found using q = %.2f. \n", qBest)

%% ADAPTIVE PLOTTING
if settings.sampling.enablePlots
     figure('Name','Sampling Plan','Color','w');

    if nDim == 1
        histogram(samplingPlan,'FaceColor',[0.2 0.4 0.8]);
        xlabel('x_1');
        ylabel('Count');
        title('1D Sampling Distribution');

    elseif nDim == 2
        scatter(samplingPlan(:,1), samplingPlan(:,2), 50, 'filled');
        xlabel('x_1'); ylabel('x_2');
        title('2D Optimal Latin Hypercube');

    elseif nDim <= 4
        % Pairwise scattermatrix
        plotmatrix(samplingPlan);
        sgtitle('Pairwise Sampling Plot');

    else
        % High-dimensional → PCA projection
        [~, score] = pca(samplingPlan);
        scatter(score(:,1), score(:,2), 35, 'filled');
        xlabel('PC1'); ylabel('PC2');
        title('PCA Projection of High-Dimensional Sampling');
    end

    grid on; box on;

    samplingPlots(samplingPlan)
end


%% SAVE SAMPLING PLAN
% Save optimized sampling plan if requested.
if settings.sampling.savePlan
    save(fullfile(dataPath, settings.sampling.fileName), ...
        'samplingPlan', 'nDim', 'nPoints')
    fprintf('Plan saved\n')
end
