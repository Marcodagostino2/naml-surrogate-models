function metrics = calculateMetrics(yTrue, yPred, relErrFloor)
arguments
    yTrue        (:,1) double
    yPred        (:,1) double
    relErrFloor  (1,1) double = 0.02
end
yTrue = yTrue(:);
yPred = yPred(:);
residuals    = yTrue - yPred;
absResiduals = abs(residuals);
rmse         = sqrt(mean(residuals.^2));
denomR2      = sum((yTrue - mean(yTrue)).^2);
r2           = 1 - sum(residuals.^2) / max(denomR2, eps);
scaleRef     = max(abs(yTrue));
denomFloor   = max(eps, relErrFloor * max(scaleRef, 1));
relDenom     = max(abs(yTrue), denomFloor);
percResiduals = 100 * residuals ./ relDenom;
absPercError  = abs(percResiduals);
metrics.rmse          = rmse;
metrics.r2            = r2;
metrics.mae           = mean(absResiduals);
metrics.maxAE         = max(absResiduals);
metrics.residuals     = residuals;
metrics.absResiduals  = absResiduals;
metrics.percResiduals = percResiduals;
metrics.mape          = mean(absPercError);
metrics.maxAPE        = max(absPercError);
metrics.denomFloor    = denomFloor;
end

