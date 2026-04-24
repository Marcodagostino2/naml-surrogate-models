function psiAug = crossCorrMatrixGEK(xTrain, xQuery, theta, p, dyDirs)
% crossCorrMatrixGEK Evaluates augmented cross-correlation between training points and queries.
    nTrain = size(xTrain, 1);
    mQuery = size(xQuery, 1);
    m = numel(dyDirs);
    
    psiAug = zeros(nTrain * (1 + m), mQuery);
    
    D = zeros(nTrain, mQuery);
    for k = 1:size(xTrain, 2)
        D = D + theta(k) * (abs(xTrain(:, k) - xQuery(:, k)').^p(k));
    end
    psi = exp(-D);
    psiAug(1:nTrain, :) = psi;
    
    for i = 1:m
        k = dyDirs(i);
        kDx = xTrain(:, k) - xQuery(:, k)';
        kAbs = abs(kDx);
        
        term = -theta(k) * p(k) * (kAbs.^(p(k)-1)) .* sign(kDx);
        
        idx = nTrain + (i-1)*nTrain + 1 : nTrain + i*nTrain;
        psiAug(idx, :) = psi .* term;
    end
end