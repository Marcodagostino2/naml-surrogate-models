function [thetaD, pD, lambdaD] = unpackCoKrigingParameters(z, d)
% unpackCoKrigingParameters  Decode co-kriging GA decision vector.
% z = [log10(thetaD_1..d), pD_1..d, lambdaD]
thetaD  = 10.^z(1:d);
pD      = z(d+1:2*d);
lambdaD = z(end);
end
