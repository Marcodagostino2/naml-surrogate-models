% ===== DATASET =====
datasetOptions.gridDataFile = 'datcomGrid.mat';
datasetOptions.nPoints      = 800;
datasetOptions.seed         = 2;

% ===== SAMPLING =====
datasetOptions.datasetMode  = "lhs";
datasetOptions.samplingPlan = [];

% ===== ACTIVE DIMENSIONS =====
datasetOptions.inNames  = {'mach', 'alpha', 'hprot'};
datasetOptions.nDim     = numel(datasetOptions.inNames);

% ===== ACTIVE OUTPUTS =====
datasetOptions.outNames = {'CA'};
datasetOptions.nOut     = numel(datasetOptions.outNames);

% ===== ACTIVE GRADIENTS =====
datasetOptions.gradNames = {};

% ===== MACH =====
datasetOptions.mach.lower   = 0.1;
datasetOptions.mach.upper   = 1.2;
datasetOptions.mach.power   = 1;
datasetOptions.mach.unit    = '[-]';
datasetOptions.mach.default = 0.8;

% ===== ANGLE OF ATTACK =====
datasetOptions.alpha.lower   = 0;
datasetOptions.alpha.upper   = 20;
datasetOptions.alpha.power   = 1;
datasetOptions.alpha.unit    = '[deg]';
datasetOptions.alpha.default = 0;

% ===== ROLL ANGLE =====
datasetOptions.phi.lower    = 0;
datasetOptions.phi.upper    = 120;
datasetOptions.phi.power    = 1;
datasetOptions.phi.unit     = '[deg]';
datasetOptions.phi.default  = 0;

% ===== ALTITUDE =====
datasetOptions.alt.lower    = 160;
datasetOptions.alt.upper    = 5160;
datasetOptions.alt.power    = 1;
datasetOptions.alt.unit     = '[m]';
datasetOptions.alt.default  = 160;

% ===== AEROBRAKES =====
datasetOptions.hprot.lower   = 0;
datasetOptions.hprot.upper   = 100;
datasetOptions.hprot.power   = 1;
datasetOptions.hprot.unit    = '[%]';
datasetOptions.hprot.default = 0;
