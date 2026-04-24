function samplingPlan = getSamplingPlan(samplingPath, nPoints, nDim, seed, samplingName)
% getSamlingPlan  Load an existing LHS sampling plan or generate a new one.
%
% If settings.dataset.samplingPlan is empty, a new plan is generated via mainSampling.
% If it is non-empty, the function tries to load it from samplingPath; if missing,
% a new plan is generated.
%
% OUTPUT
%   samplingPlan : [nPoints x nDim] samples in [0,1]^nDim
if strlength(string(samplingName)) == 0
    samplingPlan = mainSampling( ...
        nPoints=nPoints, ...
        nDim=nDim, ...
        savePlan=true, enablePlots=false, seed=seed);
    return
end

samplingPlanName = fullfile(samplingPath, samplingName);
if isfile(samplingPlanName)
    S = load(samplingPlanName, "samplingPlan");  % loads only that variable 
    samplingPlan = S.samplingPlan;
else
    samplingPlan = mainSampling( ...
        nPoints=nPoints, ...
        nDim=nDim, ...
        savePlan=true, enablePlots=false, seed=seed);
end
end

