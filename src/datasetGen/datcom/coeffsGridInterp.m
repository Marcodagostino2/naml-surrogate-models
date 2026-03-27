function [xUnit, xReal, y, dy, metadata] = coeffsGridInterp(samplingPlan, datasetOptions)
% coeffsGridInterp  Evaluate aerodynamic coefficients at unit-domain samples
%                   from a precomputed grid loaded from a .mat file.
%
% INPUTS:
%   samplingPlan   : [nPoints x nActive] unit-space samples in [0,1]^nActive.
%   datasetOptions : Simplified config. Requires: gridDataFile, inNames,
%                    outNames, gradNames, per-dim (lower/upper/power).
%                    Grid metadata (baseNames, dim layout) is read from gridDef.
% OUTPUTS:
%   xUnit     : Valid unit-domain coordinates [nPoints x nActive].
%   xReal     : Valid physical-domain coordinates [nPoints x nActive].
%   y         : Interpolated targets [nPoints x nOut].
%   dy        : Gradient targets [nPoints x nGrads] (NaN where missing).
%   metadata  : Struct with grid, geometry, and names.
arguments
    samplingPlan   (:, :) double
    datasetOptions        struct
end

%% Load Precomputed Grid
[coeffs, gridMeta, geometry] = loadGridData(datasetOptions);

%% Parse Configuration Against Grid
[inNames, outNames, baseNames, gradNames] = parseConfiguration(datasetOptions, gridMeta);
nPoints = size(samplingPlan, 1);

%% Determine Active Dimensions
% Map requested inNames to their position in the full grid
[found, gridIdx] = ismember(inNames, string(gridMeta.inNames));
assert(all(found), "coeffsGridInterp:UnknownDim", ...
    "Dimensions not found in gridDef: %s", strjoin(inNames(~found), ', '));

gridVectors      = gridMeta.gridVectors(gridIdx);
activeDimNames = string(gridMeta.inNames(gridIdx));

%% Validate Sample Dimensions
validateDimensions(samplingPlan, inNames, activeDimNames);

%% Map Unit Domain to Physical Domain
[ranges, power] = extractSpaceDef(datasetOptions, activeDimNames);
xReal = unit2real(samplingPlan, ranges, power);

%% Interpolate Coefficients
baseCoeffs = evaluateInterpolants(coeffs, baseNames, gridMeta, inNames, xReal, datasetOptions);
gradCoeffs = evaluateInterpolants(coeffs, gradNames, gridMeta, inNames, xReal, datasetOptions);

%% Compute Target Matrix (y)
[y, sdr] = assembleAeroOutputs(baseCoeffs, outNames, baseNames, xReal, activeDimNames);

%% Assemble Gradient Matrix (dy)
[dy, gradMap] = mapGradientsMatrix(gradCoeffs, gradNames, outNames, ...
    activeDimNames, nPoints, samplingPlan, ranges, power);

%% Post-Processing
[y, dy, xUnit, xReal] = filterNanRows(y, dy, samplingPlan, xReal);

%% Package Metadata
metadata = struct( ...
    'ranges',      ranges,        ...
    'power',       power,         ...
    'gridVectors', {gridVectors}, ...
    'geometry',    geometry,      ...
    'inNames',     {cellstr(inNames)},   ...
    'outNames',    {cellstr(outNames)},  ...
    'gradNames',   {cellstr(gradNames)}, ...
    'gradMap',     gradMap,       ...
    'sdr',         sdr);
end

%% =======================================================================
%  HELPER FUNCTIONS
%  =======================================================================

function [coeffs, gridMeta, geometry] = loadGridData(opts)
currentPath  = fileparts(mfilename('fullpath'));
gridDataPath = fullfile(currentPath, 'data', opts.gridDataFile);
assert(isfile(gridDataPath), "coeffsGridInterp:FileNotFound", ...
    "Grid file not found: %s", gridDataPath);
loaded   = load(gridDataPath, 'coeffs', 'gridMeta', 'geometry');
coeffs   = loaded.coeffs;
gridMeta = loaded.gridMeta;
geometry = loaded.geometry;
end


function [inNames, outNames, baseNames, gradNames] = parseConfiguration(opts, gridMeta)
inNames   = string(opts.inNames);
outNames  = string(opts.outNames);
baseNames = string(gridMeta.outNames);
gradNames = string.empty;
if isfield(opts, 'gradNames') && ~isempty(opts.gradNames)
    gradNames = string(opts.gradNames);
end
end

function validateDimensions(samplingPlan, inNames, activeDimNames)
if size(samplingPlan, 2) ~= numel(activeDimNames)
    error("coeffsGridInterp:DimMismatch", ...
        "samplingPlan has %d columns but %d active dims were resolved.", ...
        size(samplingPlan, 2), numel(activeDimNames));
end
if ~isequal(inNames, activeDimNames)
    error("coeffsGridInterp:NameMismatch", ...
        "inNames order does not match resolved active dims. Check inNames ordering.");
end
end

function interpStruct = evaluateInterpolants(coeffs, targetNames, gridMeta, inNames, xReal, opts)
% Builds griddedInterpolant on full N-D grid, queries with scattered points.
% Inactive dims are fixed to opts.(dim).fixed (fallback: opts.(dim).lower).
allDimNames = string(gridMeta.inNames);
allVectors  = gridMeta.gridVectors;
nPoints     = size(xReal, 1);

% Build per-dim query vectors [nPoints×1]: active→xReal col, inactive→fixed
queryCoords = cell(1, numel(allDimNames));
for d = 1:numel(allDimNames)
    dimName   = char(allDimNames(d));
    activeIdx = find(strcmp(inNames, dimName), 1);
    if ~isempty(activeIdx)
        queryCoords{d} = xReal(:, activeIdx);          % active: from sample
    else
        if isfield(opts.(dimName), 'fixed')
            fixedVal = opts.(dimName).fixed;
        else
            fixedVal = opts.(dimName).lower;           % fallback
        end
        queryCoords{d} = repmat(fixedVal, nPoints, 1); % inactive: constant
    end
end

interpStruct = struct();
for i = 1:numel(targetNames)
    name = char(targetNames(i));
    if ~isfield(coeffs, name)
        warning("evaluateInterpolants:MissingData", "Field '%s' not found in coeffs.", name);
        continue;
    end
    F = griddedInterpolant(allVectors, coeffs.(name), "linear", "nearest");
    interpStruct.(name) = F(queryCoords{:});   % scattered: N vettori [nPoints×1] → [nPoints×1]
end
end



function [dy, gradMap] = mapGradientsMatrix(gradCoeffs, gradNames, outNames, ...
    activeDimNames, nPoints, xUnit, ranges, power)
nGrads  = numel(gradNames);
dy      = NaN(nPoints, nGrads);
gradMap = NaN(nGrads, 2);
for g = 1:nGrads
    gName = char(gradNames(g));
    if ~isfield(gradCoeffs, gName), continue; end
    baseCoeffName = extractBefore(gName, strlength(gName));
    derVarChar    = lower(extractAfter(gName, strlength(gName) - 1));
    switch derVarChar
        case 'a', inVarName = "alpha"; conversionUnit = pi/180;
        case 'b', inVarName = "beta";  conversionUnit = pi/180;
        otherwise, inVarName = ""; conversionUnit = 1;
    end
    outIdx = find(strcmp(outNames, baseCoeffName), 1);
    inIdx  = find(strcmp(activeDimNames, inVarName), 1);
    if ~isempty(outIdx) && ~isempty(inIdx)
        lb = ranges(1, inIdx); ub = ranges(2, inIdx); p = power(inIdx);
        dxReal_dxUnit = p .* (xUnit(:, inIdx).^(p - 1)) .* (ub - lb);
        dy(:, g)      = conversionUnit * gradCoeffs.(gName) .* dxReal_dxUnit;
        gradMap(g, :) = [outIdx, inIdx];
    end
end
end
