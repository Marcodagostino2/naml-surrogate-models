function dataset = buildCfdDataset(mission, datasetOptions)
% buildCfdDataset  Reads CFD CSV results and builds the High-Fidelity dataset.
%
% This function dynamically extracts both aerodynamic targets and flight 
% condition inputs based on the provided configuration. It supports target 
% variables in two Reference Frames (SDR):
%   1. Body Frame: CA, CY, CN, Cl, Cm, Cn
%   2. Alpha-Total Frame: CXa, CYa, CZa, Cxa, Cya, Cza, SM1, SM2
%
% If Alpha-Total coefficients are requested, the function automatically 
% extracts the roll angle (phi) and computes the projections. It strictly 
% prevents mixing targets from different reference frames.
% Invalid simulations (NaNs) or samples with singular Static Margins 
% are filtered out automatically.
%
% Inputs:
%   mission        - Mission object containing general flight data
%   datasetOptions - Struct containing input/output names, bounds, and CSV mapping
%
% Outputs:
%   dataset        - Struct/Object containing normalized HF data, targets, 
%                    and metadata (including the active SDR).

arguments
    mission        Mission
    datasetOptions struct
end

%% LOAD CSV DATA
currentPath = fileparts(mfilename('fullpath'));
cfdFile     = fullfile(currentPath, "data", datasetOptions.cfdDataFile);

if ~isfile(cfdFile)
    error("buildCfdDataset:FileNotFound", "CFD data file not found: %s", cfdFile);
end

opts     = detectImportOptions(cfdFile);
cfdTable = readtable(cfdFile, opts);

%% DYNAMIC INPUT EXTRACTION
allCsvInNames = datasetOptions.csvInDimNames;
inNames       = datasetOptions.inDimNames;
activeInNames = datasetOptions.inNames;

% Map active dimensions to their corresponding CSV column names
[foundIn, locIn] = ismember(activeInNames, inNames);
if ~all(foundIn)
    missingDims = activeInNames(~foundIn);
    error("buildCfdDataset:MissingInputs", "Active input dimensions not found in inDimNames: %s", strjoin(missingDims, ', '));
end

inputCols = allCsvInNames(locIn);
xRealRaw  = cfdTable{:, inputCols};

%% STANDARD TARGET EXTRACTION (BODY FRAME)
% We assume datasetOptions.outDimNames always contains the base body names (CA, CY, CN, Cl, Cm, Cn)
targetCols = datasetOptions.csvOutDimNames;
outNames   = datasetOptions.outDimNames; 

coeffsRaw = cfdTable{:, targetCols};

% Apply conventional Body Frame signs (CA = +, CY = -, CN = +, Cl = +, Cm = -, Cn = -)
for j = 1:numel(outNames)
    if any(strcmp(outNames{j}, ["CY", "Cm", "Cn"]))
        coeffsRaw(:, j) = -coeffsRaw(:, j);
    end
end
stdCoeffs = array2table(coeffsRaw, 'VariableNames', outNames);

%% REFERENCE FRAME CHECK AND AERODYNAMIC POST-PROCESSING
activeOutNames = datasetOptions.outNames;

alphaTotNames = ["CXa", "CYa", "CZa", "Cxa", "Cya", "Cza", "SM1", "SM2"];
bodyNames     = ["CA", "CY", "CN", "Cl", "Cm", "Cn"];

isAlphaTot = any(ismember(activeOutNames, alphaTotNames));
isBody     = any(ismember(activeOutNames, bodyNames));

if isAlphaTot && isBody
    error("buildCfdDataset:MixedFrames", "Mixing Body and Alpha-Total frame targets in 'outNames' is not allowed.");
end

sdr = "body"; % Default frame
if isAlphaTot
    sdr = "alphaTot";
    
    % Dynamically retrieve the 'phi' column name from configuration
    idxPhi = find(strcmpi(datasetOptions.inDimNames, 'phi'));
    if isempty(idxPhi)
        error("buildCfdDataset:MissingPhiDef", "'phi' must be defined in datasetOptions.inDimNames to compute Alpha-Total coefficients.");
    end
    
    csvPhiName = datasetOptions.csvInDimNames{idxPhi};
    if ~ismember(csvPhiName, cfdTable.Properties.VariableNames)
        error("buildCfdDataset:MissingPhiCol", "Phi column '%s' not found in CSV.", csvPhiName);
    end
    
    phi = cfdTable.(csvPhiName); % Assuming phi is in radians
    
    % Retrieve baseline Body coefficients required for the transformation
    CA = stdCoeffs.CA; CY = stdCoeffs.CY; CN = stdCoeffs.CN;
    Cl = stdCoeffs.Cl; Cm = stdCoeffs.Cm; Cn = stdCoeffs.Cn;
    
    % Project coefficients into the Alpha-Total plane
    % Xa points backwards (CA), Ya is in the alpha-tot plane, Za is out-of-plane
    CXa = CA;
    CYa = sind(phi).*CY - cosd(phi).*CN;
    CZa = cosd(phi).*CY + sind(phi).*CN;
    Cxa = -Cl;
    Cya = sind(phi).*Cm + cosd(phi).*Cn;
    Cza = cosd(phi).*Cm - sind(phi).*Cn;
    
    % Compute Static Margins
    SM1 = zeros(size(CYa));
    SM2 = zeros(size(CZa));
    
    tol = 1e-3; % Tolerance to prevent singularities near zero force
    valid1 = abs(CYa) > tol;
    valid2 = abs(CZa) > tol;
    
    SM1(valid1)  = Cza(valid1) ./ CYa(valid1);
    SM1(~valid1) = NaN;
    
    SM2(valid2)  = Cya(valid2) ./ CZa(valid2);
    SM2(~valid2) = NaN;
    
    % Append computed coefficients to the table for dynamic extraction
    stdCoeffs.CXa = CXa; stdCoeffs.CYa = CYa; stdCoeffs.CZa = CZa;
    stdCoeffs.Cxa = Cxa; stdCoeffs.Cya = Cya; stdCoeffs.Cza = Cza;
    stdCoeffs.SM1 = SM1; stdCoeffs.SM2 = SM2;
end

%% SELECT ACTIVE TARGETS AND FILTER INVALID SAMPLES
missingActiveOuts = setdiff(activeOutNames, stdCoeffs.Properties.VariableNames);
if ~isempty(missingActiveOuts)
    error("buildCfdDataset:MissingTargets", "Active dimensions not found in computed coefficients: %s", strjoin(missingActiveOuts, ', '));
end

yRaw = stdCoeffs{:, activeOutNames};

% Filter out samples with NaNs (due to unconverged simulations or singular Static Margins)
validIdx = ~any(isnan(yRaw), 2) & ~any(isnan(xRealRaw), 2);

yHF      = yRaw(validIdx, :);
xRealHF  = xRealRaw(validIdx, :);
cfdValid = cfdTable(validIdx, :);

fprintf("  Valid CFD samples extracted: %d / %d (Reference Frame: %s)\n", size(yHF, 1), height(cfdTable), sdr);

% %% CL TRANSFORMATION: Cl -> Cl / sin(phi)
% % Fisica: Cl(mach, alpha, phi, hprot) = C_lbeta(mach, alpha, hprot) * sin(phi)
% % dove C_lbeta è una funzione LISCIA delle altre variabili.
% % Salvando C_lbeta invece di Cl il surrogate impara una funzione molto più
% % regolare, migliorando drasticamente R² su questo coefficiente.
% %
% % Al momento della predizione (CoefficientModel.get) si ri-moltiplica per sin(phi).
% %
% % GESTIONE phi ≈ 0:
% %   Per phi piccolo sin(phi) ≈ 0 e Cl ≈ 0 (simmetria assiale) quindi il
% %   rapporto è indeterminato. Usiamo una soglia phi_min_deg: sotto quella
% %   soglia C_lbeta = 0 (fisicamente corretto: il contributo è trascurabile).
% %   Il flag 'clTransformed' in metadata segnala a CoefficientModel di
% %   de-trasformare dopo la predizione.
% 
% idxCl  = find(strcmpi(activeOutNames, 'Cl'));
% idxPhi = find(strcmpi(activeInNames,  'phi'));
% 
% if ~isempty(idxCl) && ~isempty(idxPhi) && strcmpi(sdr, "body")
%     phi_deg   = xRealHF(:, idxPhi);      % phi in gradi (da CSV)
%     phi_min   = 15.0;                      % soglia [deg]: sotto non ha senso dividere
%     sin_phi   = sind(phi_deg);
% 
%     % Maschera punti con phi significativo
%     active    = abs(phi_deg) >= phi_min;
% 
%     % Calcola C_lbeta = Cl / sin(phi) solo dove phi è significativo
%     C_lbeta   = zeros(size(yHF, 1), 1);
%     C_lbeta(active)  = yHF(active,  idxCl) ./ sin_phi(active);
%     C_lbeta(~active) = 0;   % Cl ≈ 0 per phi piccolo → C_lbeta = 0
% 
%     yHF(:, idxCl) = C_lbeta;
% 
%     fprintf("  [Cl transform] Cl -> Cl/sin(phi)  (phi_min=%.1fdeg, %d/%d pts attivi)\n", ...
%         phi_min, sum(active), size(yHF,1));
%     clTransformed = true;
% else
%     clTransformed = false;
% end

%% UNIT NORMALIZATION
lb     = cell2mat(cellfun(@(d) datasetOptions.(d).lower, activeInNames, "UniformOutput", false));
ub     = cell2mat(cellfun(@(d) datasetOptions.(d).upper, activeInNames, "UniformOutput", false));
ranges = [lb; ub];
power  = cell2mat(cellfun(@(d) datasetOptions.(d).power, activeInNames, "UniformOutput", false));

xUnitHF = real2unit(xRealHF, ranges, power);

%% GEOMETRY METADATA & PACKAGING
geometryCFD = extractGeometryMetadata(cfdValid);
gridDef     = buildAerodynamicGrid(datasetOptions);
activeMask  = (gridDef.dimSizes > 1);
gridVectors = gridDef.gridVectors(activeMask);

dy = [];

dataset = constructDataset(xUnitHF, xRealHF, yHF, dy, ranges, power, ...
                           activeInNames, activeOutNames, geometryCFD, ...
                           'gridVectors', gridVectors, 'sdr', sdr);

% % Salva il flag nei metadata: CoefficientModel lo leggerà per de-trasformare
% dataset.metadata.clTransformed = clTransformed;
% if clTransformed
%     dataset.metadata.clPhiMin = phi_min;
% end

end