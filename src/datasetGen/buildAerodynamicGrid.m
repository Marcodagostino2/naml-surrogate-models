function gridDef = buildAerodynamicGrid(options)
% buildAerodynamicGrid  Initialize the full-factorial aerodynamic domain.
%
% This function extracts the grid discretization from the settings and 
% initializes the base vehicle and mission objects required by the solver.
% It acts as the single source of truth for the parameter space setup.
%
% INPUTS
%   options : Target-function struct object (contains fltcon definition).
%
% OUTPUTS
%   gridDef  : Struct containing the mathematical domain definition:
%                - dimNames       : [1 x nDim] cell array of parameter names.
%                - dimIndices     : [1 x nDim] array of parameter Dissile indices.
%                - outNames       : [1 x nOut] cell array of base coefficient names.
%                - outIndices     : [1 x nOut] array of base coefficient Datcom indices.
%                - gradDimNames   : (Optional) cell array of gradient names.
%                - gradIndices    : (Optional) array of gradient Datcom indices.
%                - dimSizes       : Array of grid sizes per dimension.
%                - dimBlock       : Struct with grouping logic for Dissile.
%                - unitSpan       : [nDim x max(p)] matrix of unit-space grid levels.
%                - realSpan       : [nDim x max(p)] matrix of physical-space grid levels.
%                - gridVectors    : Cell array of unit grid vectors.
%                - phyGridVectors : Cell array of physical grid vectors.
arguments
    options (1,1) struct
end

%% GRID SETUP
dimNames = options.inDimNames;
outNames = options.outDimNames;

if isfield(options, 'dimIndices')
    dimIndices = options.dimIndices;
else
    dimIndices = [];
end

% Fixed the copy-paste bug: checking 'outIndices' instead of 'dimIndices'
if isfield(options, 'outIndices')
    outIndices = options.outIndices;
else
    outIndices = [];
end

% Extract gradient variables if present
if isfield(options, 'gradDimNames')
    gradNames = options.gradDimNames;
else
    gradNames = {};
end

if isfield(options, 'gradIndices')
    gradIndices = options.gradIndices;
else
    gradIndices = [];
end

%% DISCRETIZATION (Real and Unit domains)
fltcon(numel(dimNames)) = struct('name', '', 'n', [], 'lower', [], 'upper', [], 'power', [], 'round', [], ...
        'maxN', [], 'symmetric', [], 'value', [], 'unit', []);
for i = 1:numel(dimNames)
    d = dimNames{i};
    fltcon(i).name      = d;
    fltcon(i).n         = options.(d).n;       
    fltcon(i).lower     = options.(d).lower;    
    fltcon(i).upper     = options.(d).upper;    
    fltcon(i).power     = options.(d).power;    
    fltcon(i).round     = options.(d).round;  
    fltcon(i).maxN      = options.(d).maxN; 
    fltcon(i).symmetric = options.(d).symmetric;
    fltcon(i).value     = options.(d).value;
    fltcon(i).unit      = options.(d).unit; 
end

% createSpan dictates the final rounded discretization grids.
[realSpan, unitSpan] = createSpan(fltcon);

%% DISSILE INPUT PACKAGING
% dimParser groups the grid evaluation into valid blocks for Dissile
[dimBlock, dimSizes, gridVectors, phyGridVectors] = dimParser(fltcon, realSpan, unitSpan);

%% PACKAGE DOMAIN DEFINITION
gridDef = struct();
gridDef.dimNames       = dimNames;
gridDef.dimIndices     = dimIndices;
gridDef.outNames       = outNames;
gridDef.outIndices     = outIndices;

% Include gradients if they exist
if ~isempty(gradNames)
    gridDef.gradNames    = gradNames;
    gridDef.gradIndices  = gradIndices;
end

gridDef.dimSizes       = dimSizes;
gridDef.dimBlock       = dimBlock;
gridDef.unitSpan       = unitSpan;
gridDef.realSpan       = realSpan;
gridDef.gridVectors    = gridVectors;
gridDef.phyGridVectors = phyGridVectors;

end
