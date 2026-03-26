function bounds = buildCoKrigingBounds(d, options)
    % Non include rho (calcolato analiticamente via GLS)
    lbLogTheta = defaultIfEmpty(options.lbLogThetaD, -5 * ones(1, d));
    ubLogTheta = defaultIfEmpty(options.ubLogThetaD,  2 * ones(1, d));
    lbP        = defaultIfEmpty(options.lbPD,         1 * ones(1, d));
    ubP        = defaultIfEmpty(options.ubPD,         2 * ones(1, d));
    lbLambda   = defaultIfEmpty(options.lbLambdaD,    1e-8);
    ubLambda   = defaultIfEmpty(options.ubLambdaD,    1e-3);
    
    bounds.lb    = [lbLogTheta(:).', lbP(:).', lbLambda];
    bounds.ub    = [ubLogTheta(:).', ubP(:).', ubLambda];
    bounds.nVars = numel(bounds.lb);
end
