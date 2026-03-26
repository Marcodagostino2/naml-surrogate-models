% CONFIG  Configuration script for main sampling setup.
%
%   This script defines all parameters required for generating and selecting
%   Latin Hypercube Sampling (LHS) plans for aerodynamic analysis. The configuration
%   is stored in the settings object, which is read by `mainSampling.m`.
%
%   The process includes:
%     - Random generation of LHS candidate plans
%     - Evaluation of space-filling quality
%     - Selection of the optimal plan
%
%   The user can configure the number of candidates, the distance metric,
%   the exponent q for φ_q, and the final selection criterion (φ_q or minimax).
%

%% SAMPLING DOMAIN
% Number of sampling points and problem dimensionality
sampling.nPoints    = 300;  % Total number of sampling points
sampling.nDim       = 3;    % Number of dimensions (e.g., mach, alpha, phi, alt, hprot)

%% LHS PLAN GENERATION
% Number of random LHS candidate plans to generate
sampling.nCandidates = 50;

% Number of itinerations
sampling.maxIterations = 100;

%% SPACE-FILLING METRIC
% Distance metric used to evaluate point dispersion
%   'euclidean' → L2 norm
%   'cityblock' → L1 norm
%   'chebyshev' → L∞ norm
sampling.metric = 'cityblock';

% Exponent q for φ_q function (only used if sortCriterion = 'phi_q' or if used preLoadedData)
sampling.q = 100;
sampling.qOptimization = [10, 50, 100]; % values of q for which optimizing sampling among

%% SELECTION CRITERION
% Criterion for selecting the optimal plan among candidates:
%   'phi_q'   → minimize φ_q (sum of inverse distances^q)
%   'maxmin' → maximize minimum pairwise distance
sampling.sortCriterion = 'maxmin';

%% PLOT 
sampling.enablePlots = true;

%% RANDOM SEED
% For reproducibility
sampling.seed = 42;

%% PRELOADED DATA OPTIONS
sampling.preLoadedData = true;

%% OUTPUT OPTIONS
% If true, saves the selected plan to a .mat file
sampling.savePlan = true;
sampling.fileName = 'lhsBestPlan.mat';

%% GEOMETRY IDENTIFIER
% Mission-specific geometry name for traceability
sampling.missionName = '2025_Orion_Portugal_October';
