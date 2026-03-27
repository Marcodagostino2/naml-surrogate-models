function dataset = buildDatcomDataset(datasetOptions)
% buildDatcomDataset  Evaluate DATCOM and package the dataset.
%
% Retrieves the LHS sampling plan, evaluates the DATCOM aerodynamic
% interpolator, structures the raw data, and packages metadata.
%
% INPUTS
%   datasetOptions : Main dataset settings struct.
%
% OUTPUTS
%   dataset        : Datcom dataset struct.

arguments
    datasetOptions struct
end

% Retrieve unit hypercube LHS plan
nPoints     = datasetOptions.nPoints;
nDim        = datasetOptions.nDim;
seed        = datasetOptions.seed;
samplingPlan = getSamplingPlan('', nPoints, nDim, seed, []);
[nPoints, nDim] = size(samplingPlan);
fprintf(" Sampling plan: %d points in %d dimensions\n", nPoints, nDim);

% Evaluate target function (DATCOM)
fprintf("  Evaluating target function \n");
[xUnit, xReal, y, dy, metadata] = coeffsGridInterp(samplingPlan, datasetOptions);

% Package dataset
dataset = constructDataset( ...
    xUnit, xReal, y, dy, ...
    metadata.ranges, metadata.power, ...
    metadata.inNames, metadata.outNames, ...
    metadata.geometry, metadata.gradMap, ...
    'gridVec', metadata.gridVectors, ...
    'sdr', metadata.sdr);

end
