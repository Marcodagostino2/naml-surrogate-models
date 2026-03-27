classdef CoefficientModel 
    % COEFFICIENTMODEL Aerodynamic surrogate model and lookup table generator.
    %
    % This class centralizes the generation of aerodynamic datasets (CFD/Datcom),
    % manages the training of Kriging or Co-Kriging surrogate models.
    %
    % It supports handling aerodynamic targets in both Body axes and Alpha-Total 
    % axes, automatically converting Alpha-Total predictions back to Body axes 
    % during inference to ensure compatibility with flight dynamics integration.
    
    properties (SetAccess = private)
        datcomDataset       Dataset           % Low-fidelity dataset wrapper
        cfdDataset          Dataset           % High-fidelity dataset wrapper
        model               SurrogateModel    % Surrogate predictor core
        geometry            struct            % Geometric reference metrics
        isValid       (1,1) logical = true    % Initialization state flag
        inNames       (1,:) string            % Active input dimension names
        outNames      (1,:) string            % Active output dimension names
        datasetMode   (1,1) string            % Operational mode: "multiFidelity", "cfd", "datcom"
    end
    
    properties (Access = private)
        stateFlags struct                     % Logic flags to suppress repetitive warnings
    end
    
    properties (Constant, Access = private)
        allInputs      = ["mach", "alpha", "beta", "phi", "alt", "hprot"];
        allOutputs     = ["CA", "CY", "CN", "Cl", "Cm", "Cn"];
        requiredInputs = ["mach", "alpha", "hprot"];
        
        % Unit references for aerodynamic dynamic simulation
        queryUnits   = struct('mach', '-', 'alpha', 'rad', 'beta', 'rad', 'phi', 'rad', 'alt', 'm', 'hprot', '-');
        datasetUnits = struct('mach', '-', 'alpha', 'deg', 'beta', 'deg', 'phi', 'deg', 'alt', 'm', 'hprot', '%');
        
        % Conversion multipliers: queryUnits -> datasetUnits
        queryToDataset = struct('mach', 1, 'alpha', 180/pi, 'beta', 180/pi, 'phi', 180/pi, 'alt', 1, 'hprot', 100);
    end
    
    methods
        function obj = CoefficientModel(options)
            % COEFFICIENTMODEL Constructor.
            % Instantiates datasets based on the selected mode, validates
            % geometries and trains the surrogate model.
            
            arguments
                options.mode        (1,1) string  = "multiFidelity"
                options.splitPerc                 = 0.85;
                options.reBuild     (1,1) logical = false
                options.reTrain     (1,1) logical = false
                options.saveDataset (1,1) logical = true
                options.saveModel   (1,1) logical = true
            end
            
            obj.datasetMode = options.mode;
            obj.stateFlags = struct('altitudeIgnored', false, 'phiIgnored', false, 'outOfBounds', struct());
            
            % Dataset Initialization & Splitting based on Mode
            switch obj.datasetMode
                case "cfd"
                    obj.cfdDataset = Dataset("cfd", 'reBuild', options.reBuild, 'saveDataset', options.saveDataset);
                    [obj.cfdDataset, train, ~] = obj.cfdDataset.split(options.splitPerc);
                    
                    primaryDataset = obj.cfdDataset;
                    
                    surrogateData = struct('x', train.xUnit, 'y', train.y);
                    obj.model = SurrogateModel("coefficient", "kriging", surrogateData, ...
                        'reTrain', options.reTrain, 'saveModel', options.saveModel);
                        
                case "datcom"
                    obj.datcomDataset = Dataset("datcom", 'reBuild', options.reBuild, 'saveDataset', options.saveDataset);
                    [obj.datcomDataset, train, ~] = obj.datcomDataset.split(options.splitPerc);
                    primaryDataset = obj.datcomDataset;

                    surrogateData = struct('x', train.xUnit, 'y', train.y);

                    % Aggiungi i gradienti se esistono
                    if isfield(train, 'dy2D') && ~isempty(train.dy2D)
                        surrogateData.dy2D = train.dy2D;
                        surrogateData.gradMap = primaryDataset.metadata.gradMap;
                    end

                    obj.model = SurrogateModel("coefficient", "kriging", surrogateData, ...
                        'reTrain', options.reTrain, 'saveModel', options.saveModel);
                        
                case "multiFidelity"
                    obj.datcomDataset = Dataset("datcom", 'reBuild', options.reBuild, 'saveDataset', options.saveDataset);
                    obj.cfdDataset    = Dataset("cfd", 'reBuild', options.reBuild, 'saveDataset', options.saveDataset);

                    [obj.cfdDataset, trainHF, ~]    = obj.cfdDataset.split(options.splitPerc);

                    primaryDataset = obj.cfdDataset; % High-fidelity used as master reference

                    if ~checkGeometryCoherence(obj.datcomDataset.metadata.geometry, obj.cfdDataset.metadata.geometry)
                        return;
                    end

                    % Aggiungiamo gradDataC per il Low-Fidelity (DATCOM)
                    surrogateData = struct(...
                        'xC', obj.datcomDataset.xUnit, 'yC', obj.datcomDataset.y, ...
                        'xE', trainHF.xUnit,           'yE', trainHF.y);
                    obj.model = SurrogateModel("coefficient", "coKriging", surrogateData, ...
                        'reTrain', options.reTrain, 'saveModel', options.saveModel);

                otherwise
                    error('Unknown dataset mode: %s. Valid options are "multiFidelity", "cfd", or "datcom".', obj.datasetMode);
            end
            
            % Extract Metadata from primary dataset
            obj.geometry    = primaryDataset.metadata.geometry;
            obj.inNames     = primaryDataset.metadata.inNames;
            obj.outNames    = primaryDataset.metadata.outNames;
            
            % Constraints Check
            for i = 1:length(obj.requiredInputs)
                if ~ismember(obj.requiredInputs(i), obj.inNames)
                    warning('CoefficientModel:MissingRequiredDim', 'Missing required dimension: %s', obj.requiredInputs(i));
                    obj.isValid = false; 
                    return;
                end
            end
            
            if ~ismember("alt", obj.inNames)
                obj.stateFlags.altitudeIgnored = true; 
            end
            if ~ismember("phi", obj.inNames)
                obj.stateFlags.phiIgnored = true; 
            end
        end

        % -----------------------------------------------------------------
        % PHYSICAL INFERENCE API
        % -----------------------------------------------------------------
        function coeffs = get(obj, mach, alpha, beta, alt, hprot, xcg)
            % GET Computes the complete 6-coefficient aerodynamic state vector.
            % Converts physical inputs, handles rotational symmetries, queries
            % the aero model, converts axes if needed (alphaTot -> body), and
            % applies spatial transport based on center of gravity.
            %
            % The method evaluates the required inputs (inNames) and routes the
            % evaluation based on the chosen angle convention:
            % - alpha-beta: ignores phi folding, uses direct angles.
            % - alphaTot-phi: computes total angle of attack and applies fin symmetry.

            if ~obj.isValid
                error('CoefficientModel:InvalidState', ...
                    'Model is invalid and cannot be used for inference.');
            end

            % Compute total angle of attack and raw roll angle from alpha and beta
            % (assuming the function returns angles in the same unit as alpha/beta)
            [alphaTot, phiRaw] = getAlphaPhi(alpha, beta);

            % Check required inputs to determine the evaluation strategy
            hasBeta = ismember("beta", obj.inNames);
            hasPhi  = ismember("phi", obj.inNames);

            % Unit conversion: query units -> dataset units
            mach  = mach  * obj.queryToDataset.mach;
            alt   = alt   * obj.queryToDataset.alt;
            hprot = hprot * obj.queryToDataset.hprot;

            nRotation = 0;
            deltaPhi  = 0;

            % Convention routing logic
            if hasBeta
                % --- Alpha-Beta Convention ---
                alphaEval = alpha * obj.queryToDataset.alpha;
                betaEval = beta * obj.queryToDataset.beta;

                % phiEval is needed if the output frame is "alphaTot" to perform
                % the inverse rotation, but NO fin folding is applied (nRotation = 0)
                phiEval  = phiRaw * obj.queryToDataset.phi;
                psiAngle = phiEval;

            elseif hasPhi
                % --- AlphaTot-Phi Convention ---
                % Map alphaTot to the "alpha" name for the dataset query
                alphaEval = alphaTot * obj.queryToDataset.alpha;
                phiEval   = phiRaw   * obj.queryToDataset.phi;

                % Fin symmetry folding on phi
                nPanels = 3;
                if isfield(obj.geometry.values, 'nPanel')
                    nPanels = obj.geometry.values.nPanel;
                end

                phiEval   = wrapTo360(phiEval);
                finAngle  = 360 / nPanels;
                nRotation = floor(phiEval / finAngle);
                deltaPhi  = nRotation * finAngle;
                psiAngle  = phiEval - deltaPhi;  % Folded evaluation angle

            else
                % --- Purely Longitudinal 1D Model ---
                alphaEval = alpha * obj.queryToDataset.alpha;
                psiAngle  = 0;
            end

            % Assemble query vector
            reqInputs = struct('mach', mach, 'alt', alt, 'hprot', hprot);

            if hasBeta
                reqInputs.alpha = alphaEval;
                reqInputs.beta  = betaEval;
            else
                reqInputs.alpha = alphaEval;
                if hasPhi
                    reqInputs.phi = psiAngle;
                end
            end

            % Extract based on obj.inNames order and clip to bounds
            xQueryPhysical = zeros(1, numel(obj.inNames));
            for i = 1:numel(obj.inNames)
                xQueryPhysical(i) = reqInputs.(obj.inNames(i));
            end

            xQueryPhysical = obj.checkBounds(xQueryPhysical);

            % Map to unit hypercube using primary dataset definition
            primaryDataset = obj.getPrimaryDataset();
            xQueryUnit     = real2unit(xQueryPhysical, primaryDataset.ranges, primaryDataset.power);

            % Inference
            yInt = obj.model.predict(xQueryUnit);

            % Force row vector [1 x nOut]
            yInt = yInt(:).';

            % Output mapping & reference frame conversion
            coeffs = nan(6, 1);
            sdrFrame = "body";

            % Detect if the high-fidelity dataset is in alpha-total reference frame
            if isfield(primaryDataset.metadata, 'sdr')
                sdrFrame = primaryDataset.metadata.sdr;
            end

            if sdrFrame == "alphaTot"
                % --- Alpha-Total to Body conversion branch ---
                % (This block works for the alpha-beta case too since 'psiAngle'
                % contains the unfolded phiRaw, allowing the correct inverse rotation)

                % Initialize alpha-total coefficients to zero
                CXa = 0; CYa = 0; CZa = 0;
                Cxa = 0; Cya = 0; Cza = 0;

                % Extract the aerodynamic coefficients that are actually present
                idx = find(obj.outNames == "CXa"); if ~isempty(idx), CXa = yInt(idx); end
                idx = find(obj.outNames == "CYa"); if ~isempty(idx), CYa = yInt(idx); end
                idx = find(obj.outNames == "CZa"); if ~isempty(idx), CZa = yInt(idx); end
                idx = find(obj.outNames == "Cxa"); if ~isempty(idx), Cxa = yInt(idx); end
                idx = find(obj.outNames == "Cya"); if ~isempty(idx), Cya = yInt(idx); end
                idx = find(obj.outNames == "Cza"); if ~isempty(idx), Cza = yInt(idx); end

                % Rebuild moment coefficients from static margins if present
                idxSm1 = find(obj.outNames == "SM1");
                if ~isempty(idxSm1)
                    SM1 = yInt(idxSm1);
                    Cza = SM1 * CYa;
                end

                idxSm2 = find(obj.outNames == "SM2");
                if ~isempty(idxSm2)
                    SM2 = yInt(idxSm2);
                    Cya = SM2 * CZa;
                end

                % Inverse rotation from AlphaTot to Body (psiAngle in degrees, valid for both)
                CA = CXa;
                CY = CYa * sind(psiAngle) + CZa * cosd(psiAngle);
                CN = -CYa * cosd(psiAngle) + CZa * sind(psiAngle);
                Cl = -Cxa;
                Cm = Cya * sind(psiAngle) + Cza * cosd(psiAngle);
                Cn = Cya * cosd(psiAngle) - Cza * sind(psiAngle);

                coeffs = [CA; CY; CN; Cl; Cm; Cn];
            else
                % --- Direct Body coefficients branch ---
                % Map outputs by name: [CA, CY, CN, Cl, Cm, Cn]
                [hasOut, locOut] = ismember(obj.allOutputs, obj.outNames);
                coeffs(hasOut) = yInt(locOut(hasOut)).';
            end

            % Restore fin symmetry rotation in the body frame (if phi folding was active)
            % If neither phi nor beta is present, it's a pure longitudinal case (CA only)
            if ~hasPhi && ~hasBeta
                coeffs(2:6) = nan;
            elseif hasPhi && nRotation > 0
                rotationMatrix = [cosd(deltaPhi) -sind(deltaPhi); ...
                    sind(deltaPhi)  cosd(deltaPhi)];
                coeffs([2, 3]) = rotationMatrix  * coeffs([2, 3]);  % Forces
                coeffs([5, 6]) = rotationMatrix' * coeffs([5, 6]);  % Moments
            end

            % XCG spatial transport to the requested reference point
            deltaXcg  = xcg - (obj.geometry.values.sim_xcgImposed ...
                - (obj.geometry.values.geom_rocket_Lnose * 1e-3));
            refLength = obj.geometry.values.geom_rocket_diameter * 1e-3;

            coeffs(4:6) = coeffs(4:6) + (deltaXcg / refLength) * [0; coeffs(3); coeffs(2)];
        end
        
        % -----------------------------------------------------------------
        % VISUAL DIAGNOSTICS
        % ----------------------------------------------------------------
        function plotDiagnostics(obj)
            % PLOTDIAGNOSTICS Generates diagnostic plots to visually analyse 
            % the continuous Surrogate Model.
            
            originalWindowStyle = get(0, 'DefaultFigureWindowStyle');
            set(0, 'DefaultFigureWindowStyle', 'docked');
            
            primary = obj.getPrimaryDataset();
            
            % Reconstruct train/test splits for plotting markers
            if obj.datasetMode == "multiFidelity"
                [~, trainLF, ~] = obj.datcomDataset.split(1.0);
                [~, trainHF, testHF] = obj.cfdDataset.split();
                baseTrainSets = [trainLF, trainHF];
                baseTestSets  = testHF;
            else
                [~, trainSet, testSet] = primary.split(0.85);
                baseTrainSets = trainSet;
                baseTestSets  = testSet;
            end
            
            for j = 1:numel(obj.outNames)
                coeffName = obj.outNames(j);
                fprintf("  Plotting %s Surrogate ...\n", coeffName);
                
                % Continuous Kriging predictor handle
                surrogateHandle = @(x) getSingleOutputPrediction(obj.model, x, j);
                
                % Structure data formatting for surrogatePlot compatibility
                slicedTrainSets = struct('xUnit', {}, 'xReal', {}, 'y', {});
                for iFid = 1:length(baseTrainSets)
                    slicedTrainSets(iFid).xUnit = baseTrainSets(iFid).xUnit;
                    slicedTrainSets(iFid).xReal = baseTrainSets(iFid).xReal;
                    slicedTrainSets(iFid).y     = baseTrainSets(iFid).y(:, j); 
                end
                
                slicedTestSets = struct('xUnit', {}, 'xReal', {}, 'y', {});
                for iFid = 1:length(baseTestSets)
                    slicedTestSets(iFid).xUnit = baseTestSets(iFid).xUnit;
                    slicedTestSets(iFid).xReal = baseTestSets(iFid).xReal;
                    slicedTestSets(iFid).y     = baseTestSets(iFid).y(:, j);
                end
                
                % Plot continuous surrogate surface
                surrogatePlot(surrogateHandle, obj.inNames, slicedTrainSets, slicedTestSets, ...
                    "outputName", coeffName, ...
                    "plotSurface", true, ...
                    "ranges", primary.ranges, ...
                    "power", primary.power);
            end
            set(0, 'DefaultFigureWindowStyle', originalWindowStyle);
        end
    end
    
    methods (Access = private)
        function ds = getPrimaryDataset(obj)
            % GETPRIMARYDATASET Returns the governing dataset based on operational mode.
            if obj.datasetMode == "datcom"
                ds = obj.datcomDataset;
            else
                ds = obj.cfdDataset; 
            end
        end
        
        function xQueryClipped = checkBounds(obj, xQuery)
            % CHECKBOUNDS Enforces physical variable boundaries to prevent 
            % unpredictable surrogate extrapolation.
            ranges = obj.getPrimaryDataset().ranges;
            xQueryClipped = max(ranges(1,:), min(ranges(2,:), xQuery));
        end
    end
end

%% HELPER FOR PLOT
function ySingle = getSingleOutputPrediction(modelObj, x, outIndex)
    % getSingleOutputPrediction Helper function to wrap multi-output prediction 
    % into a single-output function handle for surrogatePlot.
    yFull = modelObj.predict(x);
    ySingle = yFull(:, outIndex);
end
