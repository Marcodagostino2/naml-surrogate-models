function [xUnit, xReal, y, dy, metadata] = coeffsCsvInterp(datasetOptions)
% coeffsCsvInterp  Evaluate aerodynamic coefficients from a precomputed CFD CSV.
%
% Loads flight-condition inputs and aerodynamic outputs from a CSV file,
% applies reference-frame transformations, filters invalid samples, and
% normalises inputs to the unit domain.
%
% INPUTS:
%   datasetOptions : Simplified config. Requires: cfdDataFile, inNames, outNames,
%                    inDimNames/csvInDimNames, outDimNames/csvOutDimNames,
%                    per-dim (lower/upper/power).
% OUTPUTS:
%   xUnit     : Valid unit-domain coordinates     [nPoints x nActive].
%   xReal     : Valid physical-domain coordinates [nPoints x nActive].
%   y         : Extracted targets                 [nPoints x nOut].
%   dy        : Empty (no gradient data in CSV).
%   metadata  : Struct consistent with coeffsGridInterp metadata.
arguments
    datasetOptions struct
end

%% Load CSV
currentPath  = fileparts(mfilename('fullpath'));
dataFilePath = fullfile(currentPath, 'data', datasetOptions.cfdDataFile);
cfdTable     = loadCsvData(dataFilePath);

%% Extract Active Inputs
inNames  = string(datasetOptions.inNames);
outNames = string(datasetOptions.outNames);
xRealRaw = extractCsvInputs(cfdTable, inNames, datasetOptions);

%% Extract Body Coefficients (sign conventions applied)
baseNames  = string(datasetOptions.outDimNames);
bodyStruct = extractCsvBodyCoeffs(cfdTable, baseNames, datasetOptions);

%% Reference Frame Transformation and Output Assembly
[yRaw, sdr] = assembleAeroOutputs(bodyStruct, outNames, baseNames, xRealRaw, inNames);

%% Filter Invalid Samples — cfdTable filtered on same mask
[y, xReal, cfdValid] = filterNanRows(yRaw, xRealRaw, cfdTable);
fprintf("  Valid CFD samples: %d / %d (frame: %s)\n", size(y, 1), height(cfdTable), sdr);

%% Normalise to Unit Domain
[ranges, power] = extractSpaceDef(datasetOptions, inNames);
xUnit = real2unit(xReal, ranges, power);

%% Geometry from valid CFD rows
geometry = extractGeometryMetadata(cfdValid);

%% Package Metadata
dy = NaN(size(y, 1), 0);
metadata = struct( ...
    'geometry',    geometry,            ...
    'ranges',      ranges,              ...
    'power',       power,               ...
    'gridVectors', {{}},                ...
    'inNames',     {cellstr(inNames)},  ...
    'outNames',    {cellstr(outNames)}, ...
    'gradNames',   {{}},                ...
    'gradMap',     [],                  ...
    'sdr',         sdr);
end

%% =======================================================================
%  CSV-SPECIFIC HELPERS
%  =======================================================================

function cfdTable = loadCsvData(filePath)
assert(isfile(filePath), "coeffsCsvInterp:FileNotFound", ...
    "CFD CSV not found: %s", filePath);
cfdTable = readtable(filePath, detectImportOptions(filePath));
end

function xRealRaw = extractCsvInputs(cfdTable, inNames, opts)
inDimNames    = string(opts.inDimNames);
csvInDimNames = string(opts.csvInDimNames);
[found, loc]  = ismember(inNames, inDimNames);
assert(all(found), "coeffsCsvInterp:UnknownInputDim", ...
    "Active inNames not found in inDimNames: %s", strjoin(inNames(~found), ', '));
xRealRaw = cfdTable{:, csvInDimNames(loc)};
end

function bodyStruct = extractCsvBodyCoeffs(cfdTable, baseNames, opts)
% Sign convention: CY, Cm, Cn negated to match body-frame standard
CSV_SIGN_FLIP = ["CY", "Cm", "Cn"];

csvOutDimNames = string(opts.csvOutDimNames);
rawMat         = cfdTable{:, csvOutDimNames};
flipMask       = ismember(baseNames, CSV_SIGN_FLIP);
rawMat(:, flipMask) = -rawMat(:, flipMask);

bodyStruct = struct();
for k = 1:numel(baseNames)
    bodyStruct.(char(baseNames(k))) = rawMat(:, k);
end
end
