function [y, sdr] = assembleAeroOutputs(bodyStruct, outNames, baseNames, xReal, activeDimNames)
% assembleAeroOutputs  Detect reference frame, rotate if needed, extract y.
%
% Shared by coeffsGridInterp and coeffsCsvInterp. Expects bodyStruct with
% fields matching baseNames (body-frame body axes: CA,CY,CN,Cl,Cm,Cn).
ALPHA_TOT_NAMES = ["CXa","CYa","CZa","Cxa","Cya","Cza","SM1","SM2"];
nPoints    = size(xReal, 1);
isAlphaTot = any(ismember(outNames, ALPHA_TOT_NAMES));
isBody     = any(ismember(outNames, baseNames));

if isAlphaTot && isBody
    error("assembleAeroOutputs:MixedFrames", "Mixing Body and AlphaTot outputs is not allowed.");
elseif ~isAlphaTot && ~isBody
    error("assembleAeroOutputs:UnknownFrame", ...
        "outNames [%s] do not match any known reference frame.", strjoin(outNames, ', '));
end

if isAlphaTot
    sdr = "alphaTot";
    y   = computeAlphaTotOutputs(bodyStruct, outNames, xReal, activeDimNames);
else
    sdr = "body";
    y   = assembleBodyOutputs(bodyStruct, outNames, nPoints);
end
end

% -----------------------------------------------------------------------
function y = assembleBodyOutputs(bodyStruct, outNames, nPoints)
y = zeros(nPoints, numel(outNames));
for j = 1:numel(outNames)
    name = char(outNames(j));
    assert(isfield(bodyStruct, name), "assembleAeroOutputs:UnknownBodyOut", ...
        "Output '%s' not found in body coefficients.", name);
    y(:, j) = bodyStruct.(name);
end
end

% -----------------------------------------------------------------------
function y = computeAlphaTotOutputs(bodyStruct, outNames, xReal, activeDimNames)
alphaIdx = find(strcmpi(activeDimNames, 'alpha'), 1);
betaIdx  = find(strcmpi(activeDimNames, 'beta'),  1);
phiIdx   = find(strcmpi(activeDimNames, 'phi'),   1);

if ~isempty(alphaIdx) && ~isempty(phiIdx)
    phi = xReal(:, phiIdx);
elseif ~isempty(alphaIdx) && ~isempty(betaIdx)
    [~, phi] = getAlphaPhi(xReal(:, alphaIdx)', xReal(:, betaIdx)');
    phi = phi';
else
    error("assembleAeroOutputs:MissingAngles", ...
        "AlphaTot requires (alpha,phi) or (alpha,beta) in inNames.");
end

reqBody = ["CA","CY","CN","Cl","Cm","Cn"];
assert(all(isfield(bodyStruct, reqBody)), "assembleAeroOutputs:MissingBaseBody", ...
    "Full body-frame coefficients required for AlphaTot rotation.");

aTot.CXa = bodyStruct.CA;
aTot.CYa = sind(phi).*bodyStruct.CY - cosd(phi).*bodyStruct.CN;
aTot.CZa = cosd(phi).*bodyStruct.CY + sind(phi).*bodyStruct.CN;
aTot.Cxa = -bodyStruct.Cl;
aTot.Cya = sind(phi).*bodyStruct.Cm + cosd(phi).*bodyStruct.Cn;
aTot.Cza = cosd(phi).*bodyStruct.Cm - sind(phi).*bodyStruct.Cn;
aTot.SM1 = NaN(size(aTot.CYa));
aTot.SM2 = NaN(size(aTot.CZa));
v1 = abs(aTot.CYa) > 1e-3;  v2 = abs(aTot.CZa) > 1e-3;
aTot.SM1(v1) = aTot.Cza(v1) ./ aTot.CYa(v1);
aTot.SM2(v2) = aTot.Cya(v2) ./ aTot.CZa(v2);

y = zeros(size(xReal, 1), numel(outNames));
for j = 1:numel(outNames)
    name = char(outNames(j));
    assert(isfield(aTot, name), "assembleAeroOutputs:UnknownAlphaTotOut", ...
        "Unsupported AlphaTot output: %s", name);
    y(:, j) = aTot.(name);
end
end
