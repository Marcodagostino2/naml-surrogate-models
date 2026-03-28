function idx = subsampleForDisplay(n, maxPts)
% subsampleForDisplay  Return at most maxPts uniformly spaced indices from 1:n.
if n <= maxPts
    idx = 1:n;
else
    idx = round(linspace(1, n, maxPts));
end
end

