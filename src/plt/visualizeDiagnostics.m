function visualizeDiagnostics(predictor, dataset, plotSurface, outputName)
% visualizeDiagnostics  Generate surrogate diagnostic plots for the current dataset.
%
% This wrapper centralizes the plotting configuration used by mainCoefficientPrediction.
% It forwards the predictor + dataset to surrogatePlot and selects feature pairs based
% on the dataset dimensionality.
%
% Behavior:
%   - If nDim >= 2, it plots all unique 2D feature pairs.
%   - If nDim < 2, it skips 2D pairs (surrogatePlot should still handle parity/residual plots).
arguments
    predictor   (1,1) function_handle
    dataset     (1,:) struct
    plotSurface (1,1) logical = true
    outputName  (1,1) string = 'Output'
end

fprintf("Generating diagnostic plots...\n");

featureNames = dataset.dimNames;

nDim = numel(featureNames);

if nDim >= 2
    % All unique index pairs [i j], i<j.
    pairsMat = nchoosek(1:nDim, 2);     % [nPairs x 2] 
    featurePairs = mat2cell(pairsMat, ones(size(pairsMat,1),1), 2);
else
    featurePairs = {};
end

surrogatePlot(predictor, featureNames, dataset, ...
    "featurePairs", featurePairs, ...
    "plotSurface",  plotSurface, ...
    "outputName", outputName);
end


