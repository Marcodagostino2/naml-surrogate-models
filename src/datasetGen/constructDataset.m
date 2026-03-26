function dataset = constructDataset(samplingPlan, xReal, y, dy, ranges, power, inNames, outNames, geometry, gradMap,  options)
% constructDataset  Package samples, targets, and metadata into a dataset struct.
%
% This function builds the dataset "contract" used by the coefficient surrogate pipeline:
%   - xUnit: unit-hypercube inputs used to train the surrogate.
%   - xReal: physical-domain inputs corresponding to xUnit (as returned by the target function).
%   - y:    target values at each sample (may include NaNs for failed/out-of-domain evaluations).
%   - grid: optional target-function metadata (e.g., full grid / interpolant definition).
%   - dimRanges: physical ranges associated with the surrogate inputs.
%
% INPUTS
%   samplingPlan   : [nPoints x nDim] unit-domain samples (xUnit).
%   xReal          : [nPoints x nDim] physical-domain samples corresponding to xUnit.
%   y              : [nPoints x nOut] target outputs at each sample.
%   dy             : [nPoints x nGrad] target gradient outputs at each sample.
%   ranges         : [2 x nDim] matrix of physical ranges for each dimension.
%   power          : [1 x nDim] vector of power laws for each dimension.
%   inNames        : [1 x nDim] vector of dimension names.
%   outNames       : [1 x nOut] vector of dimension names.
%   geometry       : struct containing geometry information
%   gradMap        : struct containing gradient position information
%
% OUTPUT
%   dataset : struct with fields:
%       xUnit, xReal, y, functionName, dimNames, grid, dimRanges, mapRule
%
% Notes
%   dimRanges is constructed by scanning settingsFun.fltcon.dimNames and keeping only the
%   dimensions with n > 1 (i.e., varied in the target grid). The resulting order is assumed
%   to match the surrogate input ordering used by samplingPlan/settings.dataset.dimNames.
arguments
    samplingPlan        (:, :)    double
    xReal               (:, :)    double
    y                   (:, :)    double
    dy                  (:, :)    double
    ranges              (2, :)    double   
    power               (1, :)    double
    inNames             (1, :)    string
    outNames            (1, :)    string
    geometry            (1, 1)    struct
    gradMap                       double = []
    options.gridVectors           cell   = cell.empty
    options.sdr                   string = string.empty
end

nPoints = size(samplingPlan,1);

%% CORE FIELDS
dataset = struct();
dataset.xUnit         = samplingPlan;
dataset.xReal         = xReal;
dataset.y             = y;
dataset.dy            = dy;
dataset.ranges        = ranges;
dataset.power         = power;

dataset.metadata.inNames  = string(inNames);
dataset.metadata.outNames = string(outNames);
dataset.metadata.geometry = geometry;
dataset.metadata.gradMap  = gradMap;

if ~isempty(options.gridVectors)
    dataset.metadata.gridVectors = options.gridVectors;
end
if ~isempty(options.sdr)
    dataset.metadata.sdr = options.sdr;
end

%% VALID SAMPLE COUNT (row-wise)
% A sample is considered valid if all outputs are not NaN for that row. [web:261]
if size(y,1) ~= nPoints
    warning("constructDataset:SizeMismatch", ...
        "functionValues has %d rows, expected %d (same as samplingPlan).", size(y,1), nPoints);
end
validMask = ~any(isnan(y), 2);
fprintf("Dataset: %d/%d valid samples\n", sum(validMask), nPoints);
end


