function [y, varargout] = filterNanRows(y, varargin)
% filterNanRows  Remove rows where y contains NaN; apply same mask to all extra args.
%
% Usage:
%   [y, x1, x2, ...]  = filterNanRows(y, x1, x2, ...)
%
% In coeffsGridInterp: [y, dy, xUnit, xReal] = filterNanRows(y, dy, xUnit, xReal)
% In coeffsCsvInterp:  [y, xReal]            = filterNanRows(yRaw, xRealRaw)
mask       = ~any(isnan(y), 2);
numRemoved = sum(~mask);
if numRemoved > 0
    warning("filterNanRows:OutOfDomain", "Removing %d rows with NaN targets.", numRemoved);
end
y          = y(mask, :);
varargout  = cell(1, numel(varargin));
for k = 1:numel(varargin)
    varargout{k} = varargin{k}(mask, :);
end
end

