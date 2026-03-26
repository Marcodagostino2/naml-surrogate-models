% DATCOMDATASETCONFIG  Configures flight-condition dimensions for aerodynamic evaluation.
%
%   Defines all flight-condition parameters required for aerodynamic coefficient
%   computation via Dissile. Each dimension specifies discretization, mapping,
%   rounding precision, and mandatory values.
%
%   DIMENSION ORDER:
%       {mach, alpha, phi, alt, hprot}  →  {x1, x2, x3, x4, x5}
%
%   OUTPUT:
%       dataset  - Structure with flight-condition definitions
%                 .dimNames     : dimension names
%                 .(dim).maxN   : max points per Dissile call
%                 .(dim).round  : rounding precision
%                 .(dim).value  : mandatory values
%                 .(dim).n      : number of points
%                 .(dim).lower  : lower bound
%                 .(dim).upper  : upper bound
%                 .(dim).power  : power mapping (1 = uniform)
%   REVISIONS:
%       0    12/01/2026    Initial release

% ===== SAMPLING DATCOM INTERPOLATOR =====
datasetOptions.targetFunction   = @coeffsGridInterp;
datasetOptions.functionName     = "coeffsGridInterp";
datasetOptions.nPoints          = 600;
datasetOptions.seed             = 2;

% ===== INPUT SPACE DEFINITION =====
datasetOptions.datasetMode  = "lhs";
datasetOptions.samplingPlan = [];        % empty => main will generate a new LHS plan            
datasetOptions.inNames     = {'mach','alpha','phi','hprot'};
datasetOptions.nDim         = numel(datasetOptions.inNames); 

% ===== OUTPUT SPACE DEFINITION =====
datasetOptions.outNames     = {'CA'};
% datasetOptions.outNames     = {'CXa','CYa','CZa','SM1','SM2'}; 
datasetOptions.nOut         = numel(datasetOptions.outNames); 

% ===== GRADIENT SPACE DEFINITION =====
datasetOptions.gradNames     = {};

% ===== INPUT DIMENSION =====
datasetOptions.inDimNames = {'mach','alpha','phi','alt','hprot'};
% datasetOptions.inDimNames = {'mach','alpha','beta','alt','hprot'};
datasetOptions.dimIndices = [3, 2, 4, 5, 6]; % Native position in dissileMatcom output

% ===== OUTPUT DIMENSION =====
datasetOptions.outDimNames = {'CA','CY','CN','Cl','Cm','Cn'};
datasetOptions.outIndices  = [1, 3, 5, 6, 9, 13]; % Native position in dissileMatcom output 
datasetOptions.alphaTotNames = {'CXa','CYa','CZa','Cxa','Cya','Cza','SM1','SM2'};

% ===== GRADIENT DIMENSION =====
datasetOptions.gradDimNames = {'CYB','CNA','Cma','Cnb'};
datasetOptions.gradIndices  = [2, 4, 8, 12]; % Native position in dissileMatcom output 

% ===== DISSILE CONSTRAINTS =====
datasetOptions.mach.maxN  = 20;
datasetOptions.alpha.maxN = 20;
datasetOptions.beta.maxN  = [];
datasetOptions.phi.maxN   = [];
datasetOptions.alt.maxN   = [];
datasetOptions.hprot.maxN = [];

% ===== DISSILE CONSTRAINTS =====
datasetOptions.mach.symmetric  = false;
% datasetOptions.alpha.symmetric = true;
datasetOptions.alpha.symmetric = false;
datasetOptions.beta.symmetric  = false;
datasetOptions.phi.symmetric = false;
datasetOptions.alt.symmetric   = false;
datasetOptions.hprot.symmetric = false;

% ===== ROUNDING PRECISION =====
datasetOptions.mach.round  = 2;
datasetOptions.alpha.round = 1;
datasetOptions.beta.round  = 1;
datasetOptions.phi.round   = 1;
datasetOptions.alt.round   = 1;
datasetOptions.hprot.round = 4;

% ===== MANDATORY VALUES =====
datasetOptions.mach.value  = [];
% datasetOptions.alpha.value = [-1, 0, 1];
datasetOptions.alpha.value = [0, 1];
datasetOptions.beta.value  = [];
datasetOptions.phi.value   = [];
datasetOptions.alt.value   = [];
datasetOptions.hprot.value = [];

% ===== MACH SPAN =====
datasetOptions.mach.n        = 30;
datasetOptions.mach.lower    = 0.2;
datasetOptions.mach.upper    = 2.6;
datasetOptions.mach.power    = 1;
datasetOptions.mach.unit     = '[-]';

% ===== ANGLE OF ATTACK SPAN =====
datasetOptions.alpha.n        = 100;
datasetOptions.alpha.lower    = 0;
datasetOptions.alpha.upper    = 30;
datasetOptions.alpha.power    = 1;
datasetOptions.alpha.unit     = '[deg]';

% ===== SIDESLIP ANGLE SPAN =====
datasetOptions.beta.n        = 50;
datasetOptions.beta.lower    = -5;
datasetOptions.beta.upper    = 5;
datasetOptions.beta.power    = 1;
datasetOptions.beta.unit     = '[deg]';

% ===== ROLL ANGLE SPAN =====
datasetOptions.phi.n        = 10;
datasetOptions.phi.lower    = 0;
datasetOptions.phi.upper    = 120;
datasetOptions.phi.power    = 1;
datasetOptions.phi.unit     = '[deg]';

% ===== ALTITUDE SPAN =====
datasetOptions.alt.n        = 1;
datasetOptions.alt.lower    = 160;
datasetOptions.alt.upper    = 160;
datasetOptions.alt.power    = 1;
datasetOptions.alt.unit     = '[m]';

% ===== AEROBRAKES SPAN =====
datasetOptions.hprot.n        = 10;
datasetOptions.hprot.lower    = 0;
datasetOptions.hprot.upper    = 100;
datasetOptions.hprot.power    = 1;
datasetOptions.hprot.unit     = '[%]';