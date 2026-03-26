% CFDDATASETCONFIG  Configures flight-condition dimensions for aerodynamic evaluation.
%
%   Defines all flight-condition parameters required for aerodynamic coefficient
%   aquisition via CFD. Each dimension specifies discretization, mapping,
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

% ===== SAMPLING CFD =====
datasetOptions.cfdDataFile      = "CFD_SamplingPlan_output.csv";
datasetOptions.seed             = 2;

% ===== INPUT SPACE DEFINITION =====        
datasetOptions.inNames      = {'mach','alpha','phi','hprot'};
datasetOptions.nDim         = numel(datasetOptions.inNames); 

% ===== OUTPUT SPACE DEFINITION =====
datasetOptions.outNames     = {'CA'}; 
% datasetOptions.outNames     = {'CXa','CYa','CZa','Cxa','SM1','SM2'}; 
datasetOptions.nOut         = numel(datasetOptions.outNames); 

% ===== OUTPUT DIMENSION NAMES =====
datasetOptions.outDimNames = {'CA','CY','CN','Cl','Cm','Cn'};
datasetOptions.csvOutDimNames = {'cfd_cfx','cfd_cfy','cfd_cfz', ...
    'cfd_cmx','cfd_cmy','cfd_cmz'};  

% ===== INPUT DIMENSION =====
datasetOptions.inDimNames = {'mach', 'alpha', 'phi', 'alt', 'hprot'};
datasetOptions.csvInDimNames = {'sim_machNumberValue', 'sim_aoaValue', 'sim_phiValue', ...
    'sim_altitudeValue', 'geom_abk_openingPercentage'};

% ===== DISSILE CONSTRAINTS =====
datasetOptions.mach.maxN  = 20;
datasetOptions.alpha.maxN = 20;
datasetOptions.beta.maxN  = [];
datasetOptions.alt.maxN   = [];
datasetOptions.phi.maxN   = [];
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
datasetOptions.mach.lower    = 0.1;
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
datasetOptions.beta.n        = 20;
datasetOptions.beta.lower    = -2;
datasetOptions.beta.upper    = 2;
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