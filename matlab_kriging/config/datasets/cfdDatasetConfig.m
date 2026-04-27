% ===== DATASET =====
datasetOptions.cfdDataFile  = "HYDDB_v2_output.csv";
datasetOptions.seed         = 2;

% ===== INPUT SPACE =====
datasetOptions.inNames = {'mach', 'alpha', 'phi', 'alt', 'hprot'};
datasetOptions.nDim    = numel(datasetOptions.inNames);

% ===== OUTPUT SPACE =====
datasetOptions.outNames = {'CA'};
datasetOptions.nOut = numel(datasetOptions.outNames);

% ===== CSV SCHEMA - INPUTS =====
% Full list: maps standard names → CSV column names (superset of inNames)
datasetOptions.inDimNames    = {'mach',                  'alpha',        'phi',         'alt',                 'hprot'};
datasetOptions.csvInDimNames = {'sim_machNumberValue',   'sim_aoaValue', 'sim_phiValue','sim_altitudeValue',   'geom_abk_openingPercentage'};

% ===== CSV SCHEMA - OUTPUTS =====
% Body-frame coefficients available in the CSV
datasetOptions.outDimNames    = {'CA',       'CY',       'CN',       'Cl',       'Cm',       'Cn'};
datasetOptions.csvOutDimNames = {'cfd_cfx_mean',  'cfd_cfy_mean',  'cfd_cfz_mean',  'cfd_cmx_mean',  'cfd_cmy_mean',  'cfd_cmz_mean'};

% ===== MACH =====
datasetOptions.mach.lower   = 0.3;
datasetOptions.mach.upper   = 2.1;
datasetOptions.mach.power   = 1;
datasetOptions.mach.unit    = '[-]';
datasetOptions.mach.default = 0.8;

% ===== ANGLE OF ATTACK =====
datasetOptions.alpha.lower   = 0;
datasetOptions.alpha.upper   = 5;
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
datasetOptions.alt.lower    = 0;
datasetOptions.alt.upper    = 10000;
datasetOptions.alt.power    = 1;
datasetOptions.alt.unit     = '[m]';
datasetOptions.alt.default  = 160;

% ===== AEROBRAKES =====
datasetOptions.hprot.lower   = 0;
datasetOptions.hprot.upper   = 100;
datasetOptions.hprot.power   = 1;
datasetOptions.hprot.unit    = '[%]';
datasetOptions.hprot.default = 0;

