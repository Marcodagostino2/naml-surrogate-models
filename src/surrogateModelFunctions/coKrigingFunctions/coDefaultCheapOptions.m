function cheapOpt = coDefaultCheapOptions(d, options)
cheapOpt = struct();
cheapOpt.polyDegree     = defaultIfEmpty(getOpt(options,"polyDegreeC",[]), 0);

cheapOpt.lbLogTheta     = defaultIfEmpty(getOpt(options,"lbLogThetaC",[]), -5*ones(1,d));
cheapOpt.ubLogTheta     = defaultIfEmpty(getOpt(options,"ubLogThetaC",[]),  2*ones(1,d));
cheapOpt.lbP            = defaultIfEmpty(getOpt(options,"lbPC",[]),        1*ones(1,d));
cheapOpt.ubP            = defaultIfEmpty(getOpt(options,"ubPC",[]),        2*ones(1,d));
cheapOpt.lbLambda       = defaultIfEmpty(getOpt(options,"lbLambdaC",[]),   0);
cheapOpt.ubLambda       = defaultIfEmpty(getOpt(options,"ubLambdaC",[]),   0);

cheapOpt.numStarts      = getOpt(options,"numStarts",3);

cheapOpt.populationSize = getOpt(options,"populationSize",150);
cheapOpt.maxGen         = getOpt(options,"maxGen",80);
cheapOpt.maxStallGen    = getOpt(options,"maxStallGen",15);
cheapOpt.funTol         = getOpt(options,"funTol",1e-4);
cheapOpt.parallel       = getOpt(options,"parallel",true);

cheapOpt.likelihoodMode = getOpt(options,"likelihoodMode","default");
cheapOpt.seed           = getOpt(options,"seed",1);

cheapOpt.precision      = getOpt(options,"precision","double");
end

