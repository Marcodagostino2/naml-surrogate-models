function dataset = buildCfdDataset(datasetOptions)
% buildCfdDataset  Read CFD CSV results and package the High-Fidelity dataset.
%
% INPUTS
%   datasetOptions : Main dataset settings struct.
%
% OUTPUTS
%   dataset        : CFD dataset struct.

arguments
    datasetOptions struct
end

fprintf("  Evaluating target function \n");

[xUnit, xReal, y, dy, metadata] = coeffsCsvInterp(datasetOptions);

dataset = constructDataset( ...
    xUnit, xReal, y, dy, ...
    metadata.ranges, metadata.power, ...
    metadata.inNames, metadata.outNames, ...
    metadata.geometry, metadata.gradMap, ...
    'gridVec', metadata.gridVectors, ...
    'sdr',     metadata.sdr);
end
