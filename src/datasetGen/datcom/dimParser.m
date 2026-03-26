function [dimBlock, dimSizes, gridVectors, phyGridVectors] = dimParser(fltcon, realSpan, unitSpan)
% dimParser  Partition each 1D discretization into Dissile-compatible blocks.
%
% Dissile evaluations are constrained by a maximum number of points per call (maxN)
% for each dimension. This function splits the user-defined span into blocks
% that satisfy the constraint and guarantees that required values (reqVals) are
% included in every block.
%
% NEW FEATURE: Supports 'symmetric' constraints (e.g., for alpha in alpha-beta setups)
% where blocks sent to the solver MUST contain the negative counterpart of every point.
%
% INPUTS
%   fltcon   : Settings object with subtree settings.fltcon containing:
%       - (dim).n         : number of user-defined points for that dimension.
%       - (dim).maxN      : maximum number of points allowed in one block (can be empty).
%       - (dim).value     : required values to be included in every block (can be empty).
%       - (dim).symmetric : (optional) boolean. If true, forces the solver block to be symmetric.
%
%   realSpan : [k x maxN] matrix of physical discretization values (zero-padded).
%   unitSpan : [k x maxN] matrix of unit-normalized discretization values [0,1] (zero-padded).
%
% OUTPUT
%   dimBlock       : struct with one field per dimension name.
%   dimSizes       : [1 x k] array containing the total number of unique valid points per dimension.
%   gridVectors    : [1 x k] cell array containing the corresponding unit-space vectors 
%                    (deduplicated and matching the final real points) to be used in griddedInterpolant.
%   phyGridVectors : [1 x k] cell array containing the corresponding real-space vectors 
%                    (deduplicated and strictly sorted).

k        = numel(fltcon);
dimBlock = struct();

% Preallocate outputs for sizes and grid vectors
dimSizes       = zeros(1, k);
gridVectors    = cell(1, k);
phyGridVectors = cell(1, k);

for i = 1:k
    d = fltcon(i).name;
    numPts  = fltcon(i).n;
    
    % Backward-compatible check for the new 'symmetric' option
    isSym = false;
    if fltcon(i).symmetric
        isSym = true;
    end
    
    % Extract raw physical and unit spans
    spanReal = realSpan(i, 1:numPts);
    spanUnit = unitSpan(i, 1:numPts);
    
    maxN    = fltcon(i).maxN;
    reqVals = fltcon(i).value;
    
    reqVals  = reqVals(:)';   
    spanReal = spanReal(:)';  
    spanUnit = spanUnit(:)';  
    
    % Ensure base uniqueness to avoid indexing issues (keep track of indices)
    reqVals = sort(unique(reqVals), "ascend");
    [spanReal, uniqueIdx] = unique(spanReal, "stable");
    
    % Filter unitSpan using the exact same unique indices to keep physical-unit pairing perfect
    spanUnit = spanUnit(uniqueIdx);
    
    % Setup target arrays depending on the symmetry requirement
    if isSym
        evalReqVals = sort(unique([reqVals, -reqVals]), "ascend");
        evalSpanAll = sort(unique([spanReal, -spanReal, evalReqVals]), "ascend");
    else
        evalReqVals = reqVals;
        evalSpanAll = sort(unique([spanReal, reqVals]), "ascend");
    end
    
    % Single block case: if maxN is empty or large enough
    if isempty(maxN) || numel(evalSpanAll) <= maxN
        dimBlock.(d).nBlocks    = 1;
        dimBlock.(d).blockEl    = {evalSpanAll};
        dimBlock.(d).refBlockEl = {spanReal}; % Keep only requested user-span
        
        dimSizes(i)       = numel(spanReal);
        gridVectors{i}    = spanUnit(:); % Use unit coordinates for griddedInterpolant
        phyGridVectors{i} = spanReal(:); % <--- Added to prevent empty cell array error
        continue;
    end
    
    % Prevent infinite loops / crashes if constraints are contradictory
    if numel(evalReqVals) > maxN
        error("dimParser:TooManyReqVals", ...
            "Dimension '%s': required values (%d) exceed maxN constraint (%d).", ...
            d, numel(evalReqVals), maxN);
    end
    
    % Multi-block case
    remainingSpan = spanReal;     
    blockEl    = {};
    refBlockEl = {};
    
    while ~isempty(remainingSpan)
        freeSlots = maxN - numel(evalReqVals);
        newPoints = setdiff(remainingSpan, evalReqVals, "stable");
        
        if isSym
            base = [];
            for j = 1:numel(newPoints)
                p = newPoints(j);
                if ismember(p, base)
                    continue; % Already covered by a mirror
                end
                
                ptsToAdd = unique([p, -p]);
                newAdd   = setdiff(ptsToAdd, base);
                
                % Check if adding both p and -p fits in the remaining free slots
                if numel(base) + numel(newAdd) <= freeSlots
                    base = [base, newAdd];                          %#ok<AGROW>
                else
                    % Failsafe: if we can't fit even one symmetric pair, abort
                    if isempty(base)
                        error("dimParser:MaxNTooSmall", ...
                            "maxN=%d is too small to fit symmetric pairs for dimension '%s'.", maxN, d);
                    end
                    break; % Block is full
                end
            end
        else
            nTake = min(freeSlots, numel(newPoints));
            base  = newPoints(1:nTake);
        end
        
        block = sort(unique([base, evalReqVals]), "ascend");
        
        % coveredSpan grabs ONLY the requested points, ignoring the mirrored "ghost" points
        coveredSpan = intersect(remainingSpan, block, "stable");
        remainingSpan = setdiff(remainingSpan, block, "stable");  
        
        blockEl{end+1}    = block;                                  %#ok<AGROW>
        refBlockEl{end+1} = coveredSpan;                            %#ok<AGROW>
    end
    
    dimBlock.(d).nBlocks    = numel(blockEl);
    dimBlock.(d).blockEl    = blockEl;
    dimBlock.(d).refBlockEl = refBlockEl;
    
    % Reconstruct the full 1D vector of actual physical points processed IN ORDER
    fullDimReal = sort([refBlockEl{:}], 'ascend');
    
    % Find the indices of these physical points in the unique 'spanReal'
    [~, loc] = ismember(fullDimReal, spanReal);
    
    % Extract the corresponding coordinates and store them as column vectors
    dimSizes(i)       = numel(fullDimReal);
    gridVectors{i}    = spanUnit(loc)'; 
    phyGridVectors{i} = spanReal(loc)';
end
end
